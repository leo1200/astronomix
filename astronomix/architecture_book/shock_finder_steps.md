### method:

- identify shock zone:
    
    put special emphasis on filtering spurious shocks such as tangential discontinuities and contacts
    
    - identify shock direction
    - identify shock zone
- identify shock surface:
    
    tag cells with maximum compression along the shock direction and inside the shock zone
    
- calculating march value
    
    Mach number for these cells is calculated with the temperature jump across the shock zone.
    
- calculate energy dissipation
- potentially represent colliding shock zones
- look at `shock_finder` (this is 1D (I thought it is 2D) think abt 2D and 3D

### details into each step

- shock direction
    - equation:   $d_s = - \frac{\nabla T}{|\nabla T|}$
    - $\nabla T$ is computed with the second-order accurate gradient operator available in AREPO for Voronoi meshes (in paper)
    - $\nabla T$ is the Gradient of temperature
    - Method:
        - use finite difference method
            - Eg:
                - For a cell at position $(x, y, z)$:
                - $grad\_x = T(x+1) - T(x-1)$
                - $grad\_y = T(y+1) - T(y-1)$
                - $grad\_z = T(z+1) - T(z-1)$
                - The Result: $d_s$ is a unit vector (a direction arrow)
        - do this for every cell??
    - What happen after we have $d_s$?
        - we use this in the next 2 steps (at least) to find shock surface and shock zone
        - The Shock Direction ($d_s$): This is defined as $-\nabla T$ (Negative Gradient). It points from the hot gas toward the cold gas.
        - The Post-shock Direction: This is $+\nabla T$ (Positive Gradient). It points into the hot, "shocked" gas.

1. **shock zone**
    - a cell is in a shock zone if
        1. $\quad \nabla \cdot v < 0$
            - If gas is flowing *into* a cell from all sides, the divergence is negative. This means the gas is being compressed
        2. $\quad \nabla T \cdot \nabla \rho > 0$
            - checks if temperature ($T$) and density ($\rho$) are increasing in the same direction
        3. $\quad \Delta \log T \ge \left.\log \frac{T_2}{T_1}\right|_{M = M_{\min}}\quad \wedge \quad \Delta \log p \ge \left.\log \frac{p_2}{p_1}\right|_{M = M_{\min}}$
            - minimum power filter, like $T$ and $\rho$ should higher than some filter value
            - In the paper, they us $M_{\min}$ = 1.3
    - algo: loop over all cell and check above conditions
    - result (theoretically)
        - cells directly outside the shock zone in the direction of the positive temperature gradient as post-shock region
        - the corresponding cells in the direction of the negative temperature gradient are referred to as pre-shock region.
        - shock zone has a typical thickness of 3–4 cells
        
        ![image.png](attachment:4768bbb3-27c0-49f3-b536-238d2b296265:image.png)
        
        - cells inside the white contours belong to the identified ‘shock zone’;
        - black contours surround the cells that contain the reconstructed shock surface.
    
    - Algo: How to calculate these metrics of the conditions:
        - use discrete version of these calculation
        - better recheck the equations here
        - for (a)
            - $\nabla \cdot v = \frac{v_x(x+1) - v_x(x-1)}{2\Delta x} + \frac{v_y(y+1) - v_y(y-1)}{2\Delta y} + \frac{v_z(z+1) - v_z(z-1)}{2\Delta z}$
        - for (b)
            - use finite difference for $\nabla T$ and $\nabla \rho$
        - for (c)
            - pick minimum Mach number
            - calculate the actual jump in the snapshot.
                - look at the neighbor in the $d_s$ direction and the neighbor in the opposite direction.
                    
                    $\Delta \log T = \log(T_{neighbor\_hot}) - \log(T_{neighbor\_cold})$
                    

1. **Shock surface**
    - what is shock surface:
        - single layer of cells
        - so this surface will be the surface between pre-shock zone and post-shock zone
    - method:
        - Find the Peak: Within that shock zone, look for the specific cell where the gas is being "crushed" the hardest (where $\nabla \cdot v$ is the most negative)
        - info
            
            In the second paragraph of that section, the paper explains how they move from a "thick" shock zone to a "single layer" of cells. Here is the direct quote:*"Furthermore, each ray stores the velocity divergence of the cell from which it started. **If a ray traverses a cell with a smaller divergence, the ray is discarded.** ... We call these cells with **minimum velocity divergence** (i.e. maximum compression) across the shock zone the shock surface cells."*
            **Breaking down the Physics Logic**
            In fluid dynamics, **divergence (**$\nabla \cdot v$**)** measures how much a fluid is spreading out.
            • If the value is **positive**, the gas is expanding.
            • If the value is **negative**, the gas is being compressed (crushed).
            • The **"minimum"** divergence is the most negative number, which represents the point of **maximum compression**.
            
        - yeah but there are many cells belongs to a surface. so the most in what sense?
    - Algo:
        - consider a cell X that already identified as the Shock Surface
            - walking along the ray, aka follow direction of $d_s(X)$
            - As you move through the cells of the **Shock Zone**, you check the "Divergence" ($\nabla \cdot v$) of each cell
            - for example
                - Cell 1: Divergence = -10 (Squeezing)
                - Cell 2: Divergence = -50 (Squeezing harder)
                - Cell 3: Divergence = -100 (The Peak! Maximum compression)
                - Cell 4: Divergence = -30 (Squeezing is letting go)
                
                The Result: tag Cell 3 as the "Shock Surface Cell” + ignore the others for final map of surface
                
        - do this for every cell of shock zone

1. **Mach number calculation**
- Given the pre- and post-shock values, the Mach number can be calculated with any of the equations (1)–(4)

![image.png](attachment:72a2f45f-b90f-4ad3-90fa-fa06b30b9fee:image.png)

- How to have these quantities ??
    - They come directly from your **Snapshot**.
    - $\rho$ (Density) and $v$ (Velocity) are usually explicitly saved.
    - $T$ (Temperature) and $P$ (Pressure) are calculated using the **Internal Energy (**$u$**)** from the snapshot and the Ideal Gas Law:
        - $P = (\gamma - 1) \rho u$
        - $T = \frac{P \cdot \mu m_p}{\rho \cdot k_B}$
        - *(Where* $\mu$ *is the mean molecular weight,* $m_p$ *is proton mass, and* $k_B$ *is Boltzmann's constant).*

1.  **Energy dissipation**
- what: this is the result after the shock happen
- why need this step:
    - Snapshots show you the *state* (how hot it is), but the Dissipation calculation tells you the *physics* (how it got that way)
    - shock is from kinetic → thermal + radiation, this part is for thermal result
- method:
    - use equation (7) for calculating the generated thermal energy flux of a cell in the shock surface
        
        $f_{th} = \delta(M) f_{\Phi}$
        
    - Given the area of the shock surface within a shocked cell (see Section 3.1), we are
    then also able to calculate the total dissipated energy per unit time.
        
        so what is this??
        
    - equations (8) and (9) describe the thermalization efficiency at shocks without considering cosmic rays
        
        $\delta(M) =\frac{2}{\gamma(\gamma - 1) M^2 R}\left[\frac{R}{\gamma}\right]$
        
        where the density jump ( R ) is
        
        $R \equiv \frac{\rho_2}{\rho_1}\frac{(\gamma + 1) M^2}{(\gamma - 1) M^2 + 2}$
        
    
    again, how can we calculate these
    
- what will be the result here?
- how can we use the result here? like visualize it or just out put some values? if visualize then what is prefer