Dear Salvatore Orlando, dear Hans-Thomas Janka,

I'm a PhD student supervised by Tobias Buck and Nils Thuerey working on differentiable astrophysical simulations (see https://arxiv.org/abs/2607.18176) which open the door for high-dimensional PDE-constrained optimization.

With nearly 30 years of Chandra observations, I think Cas A would be a great showcase for differentiable modeling - one could optimize initial conditions / physics to closely reproduce observations (my vision would be a side-by-side movie with reconstruction and real observations + matched spectra plots etc.).

A piece I'm currently missing would be data from the 3D neutrino-driven supernova model (W15-2-cw-IIb) to get filamentary structure in the first place. Would it be possible for you to share that data? If the project sounds interesting to you, I'd also very much like to invite you to join :)

Best wishes 
Leonard

PS: I will be at the AG Meeting in Garching next week (giving a talk in the computational splinter session) and would also be happy to meet, should you also be there.

Dear Leonhard,

since I am currently on family vacation, I will not be in Garching
next week.

Your project plan is not really clear to me. My own approach usually
is to perform forward simulations self-consistently, i.e., to start
from stellar progenitors and simulate the collapse and explosion by
applying the relevant physics according to state-of-the-art knowledge.
Such models have been used by Salvatore to evolve the explosion
into the remnant stage of Cassiopeia A.

If I understand correctly, you seem to suggest some kind of
"reverse engineering", i.e. to apply some technique to "optimize"
the initial conditions such that some observations are reproduced,
for example the Cas A supernova remnant
(which, by the way, is meanwhile studied by high-resolution JWST
observations).

This is an approach that Salvo has also taken for some problems
(SN 1987A, Cas A), but it is not my own preference. Generally, I am
quite sceptical about such an approach, because it does not solve
the fundamental question which physics leads to the constructed
optimal initial state. Actually, it leaves the big questions open
what the initial state really means and whether it is an unambiguous
solution. In the core collapse problem: Actually what *is* the
initial state, what do you consider as the intial state? Is it
the star before collapse? Or the explosion at its beginning, let's
say 1 second after core bounce? Or the stellar explosion at one day
later??? Note that the physics in all these evolution stages is
drastically different with respect to plasma equation of state,
radiation effects, radioactive energy input, nuclear reactions,
ionisation physics etc.

Of course, we can share data from my student Annop Wongwathanarat's
model W15-2-cw-IIb, of which some 3D outputs at certain times are
stored in our Garching Core-Collapse Supernova Archive
https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/
However, without a detailed understanding of the planned usage of
these data, it is unclear to me how well they can serve your needs.
The model has good sides but also shortcomings, it's more than 10
years old.
Salvatore has evolved this early explosion model (from < one day)
with his PLUTO code to the stage of the supernova remnant nowdays
and in the future, taking into account a lot of the physics that
plays a role during this long-term evolution.
Based on these data you may choose more suitable later states than
provided by Annop Wongwathanarat's early explosion models.

With best regards,

Thomas

There is one more thing to add:
The comparison of models and observations of supernova remnants
is by no means trivial. The observed properties are often projections
and defined by the detailed radiation conditions, dependent on local
density, temperature, composition, ionization levels.
In many cases the true morphology of the remnant cannot be
unambiguously identified on grounds of observed emission and
information about the physical state is also incomplete, because
the observations do not cover all relevant frequences and spatial
regions.
This, in my opinion, is a great handicap for any approach that
strives for determining optimized initial conditions on ground of
observational data.

Cheers,
Thomas