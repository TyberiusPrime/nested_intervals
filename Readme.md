A library to deal with interval sets.

We can
- [x] build them
- [x] check for overlap with a query range -> has_overlap
- [x] query for overlapping with a query range -> filter_to_overlapping
- [x] query for overlapping with a query set -> filter_to_overlapping_multiple
- [x] query for non-overlapping with a query set -> filter_to_non_overlapping
- [x] check for internally overlapping -> any_overlapping
- [x] remove duplicate intervals (start&stop!)-> remove_duplicates
- [x] remove duplicate intervals and complain about non-duplicate overlapping ->
  remove_duplictes & any_overlapping
- [x] merge internally overlapping by joining them -> merge_hull
- [ ] merge internally overlapping by intersecting them? what does than even mean for
  nested intervals?
- [x] merge internally overlapping by dropping them -> merge_drop
- [ ] find the closest interval to a point
- [x] find the interval with the closest start to the left of a point -> find_closest_start_left
- [x] find the interval with the closest start to the right of a point -> find_closest_start_right
- [ ] find the interval with the closest end to the left of a point
- [ ] find the interval with the closest end to the right of a point
- [x] calculate the units covered by the intervals -> covered_units() 
- [x] find the mean size of the intervals
- [x] build the union of n interval sets
- [ ] substract two interval sets 
- [ ] keep only those overlapping with n other sets
- [ ] remove those overlapping with more than n other sets
- [x] invert an interval set (given outer borders)
- [ ] intersect to interval sets
- [ ] split intervalsets into non-overlapping subsets
- [ ]
- [ ]
