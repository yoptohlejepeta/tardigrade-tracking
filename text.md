

orig 

Selected observables and their distributions were measured on segmented samples
to describe mostly shape properties of tardigrades and metrices that capture
the variation of these properties in time. These are, center-of-mass (COM) and
area of each segmented object, basic shape descriptors: sphericity, solidity,
eccentricity, ellipse axes and their distributions.

Following approach has been adopted to obtain reliable statistics that reflect
current pros and cons of the segmentation process. These are for example
overlaid tardigrades in the image that are segmented as one object or one
tardigrade split into two separate objects. Both cases, change number of
objects in the snapshot and more importantly negatively influence calculation
of observables. According to this, we first identified current number of
objects on the snapshot and obtained its variation in time. Then, together with
biologist, we manually set the “proper” number of objects in each processed
image. Finally, we calculated the overall statistics using only those snapshots
that selected same number of tardigrades. Moreover, to track individual
tardigrade in time, we observed maximum displacement of COM of each object
between two snapshots and rejected those whose displacement is larger than half
of the size of the object itself. Such approach guaranteed that we always track
the same tardigrade in the video. Variation of number of objects and its
distribution is shows in Figure SXXX in Supporting information.


changed

...

Following approach has been adopted to obtain reliable statistics that reflect
current pros and cons of the segmentation process. These are for example
overlaid tardigrades in the image that are segmented as one object or one
tardigrade split into two separate objects. Both cases, change number of
objects in the snapshot and more importantly negatively influence calculation
of observables. According to this, we first identified current number of
objects on the snapshot and obtained its variation in time. Then, together with
biologist, we manually set the "proper" number of objects in each processed
image. Finally, we calculated the overall statistics using only those snapshots
that selected same number of tardigrades. Moreover, to track individual
tardigrade in time, we used linear sum assignment to match objects between
consecutive frames based on their centroid distances, which ensured optimal
pairing and consistent tracking of the same tardigrade throughout the video.
Variation of number of objects and its distribution is shows in Figure SXXX in
Supporting information.



- [x] tlustsi caru, jinou barvu labelu (zelena)

- [x] lepsi vetsi obrazky, watershed hranice + labely (nekde kde je to vsude dobre) ( ulozit figures/si/workflow/images )

- [x] c2, c3, c4, c6 - sloupce id snimku a pocet objektu

doplnit tabulku

