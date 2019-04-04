A library to deal with interval sets.

We can
- [x] build them
- [x] check for overlap with a query range -> has_overlap
- [x] query for overlapping with a query range -> query_overlap
- [x] query for overlapping with a query set -> query_overlap_multiple
- [x] query for non-overlapping with a query set -> query_non_overlap
- [x] check for internally overlapping -> any_overlapping
- [x] remove duplicate intervals (start&stop!)-> remove_duplicates
- [x] remove duplicate intervals and complain about non-duplicate overlapping ->
  remove_duplictes & any_overlapping
- [ ] merge internally overlapping by joining them
- [ ] merge internally overlapping by intersecting them?
- [ ] merge internally overlapping by dropping them
- [ ] find the closest interval to a point
- [ ] find the closest interval to the left of a point
- [ ] find the closest interval to the right of a point
- [ ] calculate the units covered by the intervals
- [ ] find the mean size of the intervals
- [ ] build the union of n interval sets
- [ ] substract two interval sets 
- [ ] keep only those overlapping with n other sets
- [ ] remove those overlapping with more than n other sets
- [ ] invert an interval set (given outer borders)
- [ ] intersect to interval sets
- [ ] split intervalsets into non-overlapping subsets
- [ ]
- [ ]
