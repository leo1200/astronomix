"""
Problem sampling module for multi-GPU training.

This module provides the ProblemSampler class for managing per-epoch sampling
of training problems. Supports multiple sampling strategies:

1. Random sampling: Each epoch randomly samples from available problems
2. Fixed subset: Trains on equally-spaced fixed subset every epoch
3. Subset sampling: Randomly samples from a fixed pool each epoch

The subset sampling mode enables training on a smaller random subset per epoch
while maintaining a deterministic pool of problems, increasing diversity while
keeping computation manageable.

Example usage:
    # Random sampling (default)
    sampler = ProblemSampler(
        problem_descriptors=descriptors,
        problem_counts={"mhd_blast": 30},
    )
    
    # Fixed subset (same problems every epoch)
    sampler = ProblemSampler(
        problem_descriptors=descriptors,
        problem_counts={"mhd_blast": 30},
        use_fixed_subset=True,
    )
    
    # Subset sampling (random from fixed pool)
    sampler = ProblemSampler(
        problem_descriptors=descriptors,
        problem_counts={"mhd_blast": 120},  # Pool size
        use_fixed_subset=True,
        subset_size=10,  # Per-epoch sample size
    )
"""

import logging
import random
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _descriptor_problem_name(problem_descriptor) -> str:
    """Resolve the canonical problem name from descriptor variants.
    
    Args:
        problem_descriptor: Problem descriptor object with either 'name' or 
                          'problem_name' attribute.
    
    Returns:
        The problem name string.
        
    Raises:
        AttributeError: If descriptor has neither 'name' nor 'problem_name'.
    """
    if hasattr(problem_descriptor, "name"):
        return problem_descriptor.name
    if hasattr(problem_descriptor, "problem_name"):
        return problem_descriptor.problem_name
    raise AttributeError(
        f"Problem descriptor has no 'name' or 'problem_name': {problem_descriptor}"
    )


class ProblemSampler:
    """Manages per-epoch sampling of problems based on configured counts.
    
    Supports three modes:
    1. Random sampling: Each epoch randomly samples from available problems (default)
    2. Fixed subset: Trains on equally-spaced fixed subset every epoch (excludes extrema)
    3. Subset sampling: Randomly samples subset_size from fixed pool each epoch
    
    Attributes:
        problem_descriptors: List of all available problem descriptors
        problem_counts: Dict mapping problem names to pool sizes
        use_fixed_subset: If True, use fixed pool instead of full random sampling
        subset_size: Optional per-epoch sample size (random from fixed pool)
        problem_pairs_map: Dict mapping problem names to descriptor lists
        fixed_subsets: Dict mapping problem names to fixed pool (when use_fixed_subset=True)
    """

    def __init__(
        self,
        problem_descriptors: List,
        problem_counts: Dict[str, int],
        use_fixed_subset: bool = False,
        subset_size: Optional[int] = None,
    ):
        """Initialize the problem sampler.
        
        Args:
            problem_descriptors: List of all available problem descriptors
            problem_counts: Dict mapping problem names to counts (pool size if 
                          use_fixed_subset=True, otherwise per-epoch count)
            use_fixed_subset: If True, pre-select equally-spaced fixed pool
            subset_size: Optional per-epoch sample size. If specified, must have
                        use_fixed_subset=True. Enables subset sampling mode.
        
        Raises:
            ValueError: If subset_size is specified without use_fixed_subset=True
        """
        self.problem_descriptors = problem_descriptors
        self.problem_counts = problem_counts
        self.use_fixed_subset = use_fixed_subset
        self.subset_size = subset_size
        
        # Validate subset_size usage
        if subset_size is not None and not use_fixed_subset:
            raise ValueError(
                "subset_size can only be used with use_fixed_subset=True. "
                "Use --fixed-problems when specifying --subset."
            )

        # Build lookup: problem name -> list of descriptors with that name
        self.problem_pairs_map: Dict[str, List] = {}
        for descriptor in problem_descriptors:
            problem_name = _descriptor_problem_name(descriptor)
            if problem_name not in self.problem_pairs_map:
                self.problem_pairs_map[problem_name] = []
            self.problem_pairs_map[problem_name].append(descriptor)

        # If using fixed subset, pre-select equally-spaced problems
        self.fixed_subsets: Dict[str, List] = {}
        if self.use_fixed_subset:
            for problem_name, count in self.problem_counts.items():
                self.fixed_subsets[problem_name] = self._select_fixed_subset(
                    problem_name, count
                )
            
            # Log initialization with appropriate mode description
            if self.subset_size is not None:
                logger.info(
                    f"ProblemSampler initialized in SUBSET SAMPLING mode"
                )
                for problem_name in sorted(self.fixed_subsets.keys()):
                    pool_size = len(self.fixed_subsets[problem_name])
                    logger.info(
                        f"  {problem_name}: pool={pool_size} (equally-spaced), "
                        f"per-epoch={self.subset_size} (random from pool)"
                    )
            else:
                logger.info(
                    f"ProblemSampler initialized in FIXED mode: {self.problem_counts}"
                )
                for problem_name in sorted(self.fixed_subsets.keys()):
                    logger.info(
                        f"  {problem_name}: {len(self.fixed_subsets[problem_name])} "
                        f"equally-spaced problems (excluding extrema)"
                    )
        else:
            logger.info(
                f"ProblemSampler initialized in RANDOM mode: {self.problem_counts}"
            )

    def _select_fixed_subset(self, problem_name: str, count: int) -> List:
        """Select equally-spaced subset of problems, excluding extrema.
        
        Uses numpy.linspace to select indices that are evenly distributed
        across the dataset, excluding the first and last indices (extrema).
        
        Args:
            problem_name: Name of the problem type
            count: Number of problems to select for the fixed pool
            
        Returns:
            List of selected descriptors
        """
        available_descriptors = self.problem_pairs_map[problem_name]
        total_available = len(available_descriptors)
        
        if total_available <= 2:
            # Edge case: not enough problems to exclude extrema
            logger.warning(
                f"Problem '{problem_name}' has only {total_available} samples. "
                f"Using all available samples."
            )
            return available_descriptors[:count]
        
        # Calculate equally-spaced indices excluding extrema
        # linspace from 1 to (total_available-1), excluding endpoint
        if count >= total_available - 2:
            # If requesting more than available (minus extrema), use all except extrema
            selected_indices = list(range(1, total_available - 1))
        else:
            # Use linspace to get equally-spaced indices
            selected_indices = np.linspace(
                1, total_available - 1, count, endpoint=False, dtype=int
            )
        
        selected_descriptors = [available_descriptors[i] for i in selected_indices]
        
        logger.debug(
            f"Fixed subset for {problem_name}: selected indices {list(selected_indices)} "
            f"from range [1, {total_available-1})"
        )
        
        return selected_descriptors

    def sample_epoch(self) -> List:
        """Sample descriptors for this epoch based on configured counts.
        
        Behavior depends on initialization mode:
        - Random mode: Randomly samples problems each epoch
        - Fixed mode (no subset_size): Returns same fixed subset every epoch
        - Subset sampling mode: Randomly samples subset_size from fixed pool

        Returns:
            List of sampled descriptors for this epoch.
        """
        if self.use_fixed_subset:
            # Fixed subset or subset sampling mode
            sampled_pairs = []
            
            for problem_name in self.problem_counts.keys():
                fixed_pool = self.fixed_subsets[problem_name]
                
                if self.subset_size is not None:
                    # Subset sampling: randomly sample from fixed pool
                    sample_size = min(self.subset_size, len(fixed_pool))
                    sampled = random.sample(fixed_pool, k=sample_size)
                    sampled_pairs.extend(sampled)
                else:
                    # Fixed mode: return entire fixed pool
                    sampled_pairs.extend(fixed_pool)
            
            return sampled_pairs
        
        # Random sampling mode (original behavior)
        sampled_pairs = []
        for problem_name, count in self.problem_counts.items():
            if problem_name not in self.problem_pairs_map:
                raise ValueError(
                    f"Problem '{problem_name}' not found. Available: "
                    f"{list(self.problem_pairs_map.keys())}"
                )

            available_descriptors = self.problem_pairs_map[problem_name]
            # Sample with replacement if count > available descriptors
            sampled = random.choices(available_descriptors, k=count)
            sampled_pairs.extend(sampled)

        return sampled_pairs
