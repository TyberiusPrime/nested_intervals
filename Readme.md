A library to deal with interval sets.

We can
- [x] build them
- [x] check for overlap with a query range -> has_overlap
- [x] query for overlapping with a query range -> query_overlapping
- [x] query for overlapping with a query set -> filter_to_overlapping_multiple
- [x] query for non-overlapping with a query set -> filter_to_non_overlapping
- [x] check for internally overlapping -> any_overlapping
- [x] remove duplicate intervals (start&stop!)-> remove_duplicates
- [x] remove duplicate intervals and complain about non-duplicate overlapping ->
  remove_duplictes & any_overlapping
- [x] merge internally overlapping by joining them -> merge_hull
- [x] merge internally overlapping by dropping them -> merge_drop
- [x] split internally overlapping into non-overlapping subsets (ie. [10..100, 20..30] becomes
  [10..20, 20..30, 30..100]:  merge_split()
- [x] invert an interval set (given outer borders) -> invert()
- [x] find the interval with the closest start -> find_closest_start
- [x] find the interval with the closest start to the left of a point -> find_closest_start_left
- [x] find the interval with the closest start to the right of a point -> find_closest_start_right

- [x] calculate the units covered by the intervals -> covered_units() 
- [x] find the mean size of the intervals -> mean_interval_size()
- [x] build the union of n interval sets -> union()
- [x] substract two interval sets  -> substract() (should this be named difference?)
- [x] keep only those overlapping with n other sets -> filter_to_overlapping_k_others
- [x] remove those overlapping with more than n other sets ->filter_to_non_overlapping_k_others
- 
We currently can not
- [ ] find the interval with the closest end
- [ ] find the interval with the closest end to the left of a point //going be expensive O(n/2)
- [ ] find the interval with the closest end to the right of a point //going be
  expensiv O(n/2)

- [ ] intersect two interval sects(ie. covered units in both sets)
- [ ] intersect in more than interval sects(ie. covered units in multiple sets, possibly
  applying a 'k')

- [ ] merge internally overlapping by intersecting them? what does than even mean
  for nested sets?
  
