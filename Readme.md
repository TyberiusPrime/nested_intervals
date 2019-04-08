nested_intervals
-----------------
[![Crates.io](https://img.shields.io/crates/d/nested_intervals.svg)](https://crates.io/crates/nested_intervals)
[![docs.rs](https://docs.rs/nested_intervals/badge.svg)](https://docs.rs/nested_intervals/badge.svg)


This crate deals with interval sets which are lists of 
[Ranges](https://doc.rust-lang.org/std/ops/struct.Range.html) that may be both
overlapping and nested.

The implementation is based on nested containment lists as proposed 
by [Alekseyenko et al. 2007](https://www.ncbi.nlm.nih.gov/pubmed/17234640),
which offers the same big-O complexity s interval trees (O(n * log(n)) construction,
O(n + m) queries). The construction of the query data structure is lazy and only happens
the first time a method relying on it is called.

Each interval has a vec of u32 ids attached, which allows linking back the results to
other data structures.

Full [documentation at docs.rs](https://docs.rs/nested_intervals)
Source at [GitHub](https://github.com/TyberiusPrime/nested_intervals)


Example
--------

Code example:
```
  fn test_example() {
        let intervals = vec![0..20, 15..30, 50..100];
        let mut interval_set = IntervalSet::new(&intervals);
        assert_eq!(interval_set.ids, vec![vec![0], vec![1], vec![2]]); // automatic ids, use new_with_ids otherwise
        let hits = interval_set.query_overlapping(10..16);
        assert_eq!(hits.intervals, [0..20, 15..30]);
        let merged = hits.merge_hull();
        assert_eq!(merged.intervals, [0..30]);
        assert_eq!(merged.ids, vec![vec![0,1]]);
    }
```


Functionality
-------------

- [x] check for overlap with a query range ->
  [has_overlap](http://docs.rs/nested_intervals/struct.IntervalSet.html#method.has_overlap)
- [x] query for overlapping with a query range ->
  [query_overlapping](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.query_overlapping)
- [x] query for overlapping with a query set ->
  [filter_to_overlapping_multiple](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.filter_to_overlapping_multiple)
- [x] query for non-overlapping with a query set ->
  [filter_to_non_overlapping](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.filter_to_non_overlapping)
- [x] check for internally overlapping ->
  [any_overlapping](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.any_overlapping)
- [x] check for internally nested ->
  [any_nested](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.any_nested)
- [x] remove duplicate intervals (start&stop!)->
  [remove_duplicates](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.remove_duplicates)
- [x] remove duplicate intervals and complain about non-duplicate overlapping ->
  [remove_duplictes
  ](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.remove_duplictes 
  & 
  [any_overlapping](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.any_overlapping)
- [x] remove empty intervals ->
  [remove_empty](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.remove_empty)
- [x] merge internally overlapping by joining them ->
  [merge_hull](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.merge_hull)
- [x] merge internally overlapping by dropping them ->
  [merge_drop](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.merge_drop)
- [x] split internally overlapping into non-overlapping subsets (ie. [10..100, 20..30] becomes
  [10..20, 20..30, 30..100] ->
  [merge_split](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.merge_split)
- [x] invert an interval set (given outer borders) ->
  [invert](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.invert)
- [x] find the interval with the closest start ->
  [find_closest_start](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.find_closest_start)
- [x] find the interval with the closest start to the left of a point ->
  [find_closest_start_left](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.find_closest_start_left)
- [x] find the interval with the closest start to the right of a point ->
  [find_closest_start_right](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.find_closest_start_right)

- [x] calculate the units covered by the intervals ->
  [covered_units](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.covered_units)
- [x] find the mean size of the intervals ->
  [mean_interval_size](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.mean_interval_size)
- [x] build the union of n interval sets -> union
- [x] substract two interval sets  ->
  [filter_to_non_overlapping](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.filter_to_non_overlapping)
- [x] keep only those overlapping with n other sets ->
  [filter_to_overlapping_k_others](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.filter_to_overlapping_k_others)
- [x] remove those overlapping with more than n other sets -> 
  [filter_to_non_overlapping_k_others](https://docs.rs/nested_intervals/struct.IntervalSet.html#method.filter_to_non_overlapping_k_others)

Not (yet) supported
--------------------
We currently can not
- [ ] find the interval with the closest end
- [ ] find the interval with the closest end to the left of a point //going be expensive O(n/2)
- [ ] find the interval with the closest end to the right of a point //going be
  expensiv O(n/2)

- [ ] intersect two interval sects (ie. covered units in both sets)
- [ ] intersect more than two interval sects (ie. covered units in multiple sets, possibly
  applying a 'k' threshold)

- [ ] merge internally overlapping by intersecting them? What does than even mean
  for nested sets?
  
