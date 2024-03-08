use itertools::Itertools;
use std::cell::RefCell;
use std::cmp::{max, min, Ordering};
use std::collections::HashMap;
use std::ops::Range;
use superslice::*;

trait FilterByBools<T> {
    fn filter_by_bools(&self, keep: &[bool]) -> Vec<T>;
}

impl<T> FilterByBools<T> for Vec<T>
where
    T: Clone,
{
    fn filter_by_bools(&self, keep: &[bool]) -> Vec<T> {
        if self.len() != keep.len() {
            panic!("v and keep had unequal length");
        }
        self.iter()
            .enumerate()
            .filter(|(idx, _value)| keep[*idx])
            .map(|(_idx, value)| value.clone())
            .collect()
    }
}

/// the integer type for the interval's ranges.
/// e.g. u32, u64
pub trait Rangable: num::PrimInt {}

impl Rangable for u16 {}
impl Rangable for u32 {}
impl Rangable for u64 {}
impl Rangable for u128 {}

impl Rangable for i16 {}
impl Rangable for i32 {}
impl Rangable for i64 {}
impl Rangable for i128 {}

/// IntervalSetGeneric
///
/// A collection of Range<Rangable> and associated ids (u32).
///
/// If no ids are supplied, default ones will be provided.
/// Merging functions return sorted ids
/// Little assumption is placed on the intervals, they may
/// overlap and nest. They must be start <= end though.
///
/// Our ranges are like Rust ranges, left closed, right open.
///
/// Internal storage is sorted by (start, -end), which is enforced
/// at construction time.
#[derive(Debug)]
pub struct IntervalSetGeneric<T: Rangable + std::fmt::Debug> {
    intervals: Vec<Range<T>>,
    ids: Vec<Vec<u32>>,
    root: RefCell<Option<IntervalSetEntry>>,
}

/// An IntervalSetGeneric<u32>
pub type IntervalSet = IntervalSetGeneric<u32>;

/// IntervalSetEntry
///
/// Used internally to build the nested containment list
/// Note that we do not reference the Intervals directly
/// but hide them behind an index into IntervalSet.intervals (and .ids)
/// thus avoiding all lifetime discussions but incuring a bit of
/// runtime overhead. On the otherhand, we'd need two pointers
/// to the .intervals and the .ids, or replace those by tuples
///
#[derive(Debug)]
struct IntervalSetEntry {
    no: i32,
    children: Vec<IntervalSetEntry>,
}

impl<T: Rangable + std::fmt::Debug> Clone for IntervalSetGeneric<T> {
    fn clone(&self) -> IntervalSetGeneric<T> {
        IntervalSetGeneric {
            intervals: self.intervals.clone(),
            ids: self.ids.clone(),
            root: self.root.clone(),
        }
    }
}

impl Clone for IntervalSetEntry {
    fn clone(&self) -> IntervalSetEntry {
        IntervalSetEntry {
            no: self.no,
            children: self.children.clone(),
        }
    }
}

trait IntervalCollector<T: Rangable + std::fmt::Debug> {
    fn collect(&mut self, iset: &IntervalSetGeneric<T>, no: u32);
}

struct VecIntervalCollector<T: Rangable> {
    intervals: Vec<Range<T>>,
    ids: Vec<Vec<u32>>,
}

impl<T: Rangable + std::fmt::Debug> VecIntervalCollector<T> {
    fn new() -> VecIntervalCollector<T> {
        VecIntervalCollector {
            intervals: Vec::new(),
            ids: Vec::new(),
        }
    }
}

impl<T: Rangable + std::fmt::Debug> IntervalCollector<T> for VecIntervalCollector<T> {
    fn collect(&mut self, iset: &IntervalSetGeneric<T>, no: u32) {
        self.intervals.push(iset.intervals[no as usize].clone());
        self.ids.push(iset.ids[no as usize].clone());
    }
}
struct TagIntervalCollector {
    hit: Vec<bool>,
}

impl TagIntervalCollector {
    fn new<T: Rangable + std::fmt::Debug>(iset: &IntervalSetGeneric<T>) -> TagIntervalCollector {
        TagIntervalCollector {
            hit: vec![false; iset.len()],
        }
    }
}

impl<T: Rangable + std::fmt::Debug> IntervalCollector<T> for TagIntervalCollector {
    fn collect(&mut self, _iset: &IntervalSetGeneric<T>, no: u32) {
        self.hit[no as usize] = true;
    }
}

/// nclists are based on sorting the intervals by (start, -end)
#[allow(clippy::needless_return)]
fn nclist_range_sort<T: Rangable>(a: &Range<T>, b: &Range<T>) -> Ordering {
    if a.start < b.start {
        return Ordering::Less;
    } else if a.start > b.start {
        return Ordering::Greater;
    } else if a.end > b.end {
        return Ordering::Less; // the magic trick to get contained intervals
    } else if a.end < b.end {
        return Ordering::Greater;
    } else {
        return Ordering::Equal;
    }
}

#[derive(Debug)]
pub enum NestedIntervalError {
    ///a negative interval was passed, ie. stop < start
    NegativeInterval,
    ///intervals and ids had differing lengths
    IntervalIdMisMatch,
}

impl<T: Rangable + std::fmt::Debug> IntervalSetGeneric<T> {
    /// Create an IntervalSet without supplying ids
    ///
    /// ids will be 0..n in the order of the *sorted* intervals
    pub fn new(intervals: &[Range<T>]) -> Result<IntervalSetGeneric<T>, NestedIntervalError> {
        for r in intervals {
            if r.start >= r.end {
                return Err(NestedIntervalError::NegativeInterval);
            }
        }
        let mut iv = intervals.to_vec();
        iv.sort_unstable_by(nclist_range_sort);
        let count = iv.len();
        Ok(IntervalSetGeneric {
            intervals: iv,
            ids: (0..count).map(|x| vec![x as u32]).collect(),
            root: RefCell::new(None),
        })
    }

    /// Create an IntervalSet
    ///
    /// Ids may be non-unique
    /// This consumes both the intervals and ids
    /// which should safe an allocation in the most common use case
    pub fn new_with_ids(
        intervals: &[Range<T>],
        ids: &[u32],
    ) -> Result<IntervalSetGeneric<T>, NestedIntervalError> {
        let vec_ids = ids.iter().map(|x| vec![*x]).collect::<Vec<Vec<u32>>>();
        Self::new_with_ids_multiple(intervals, vec_ids)
    }
    pub fn new_with_ids_multiple(
        intervals: &[Range<T>],
        ids: Vec<Vec<u32>>,
    ) -> Result<IntervalSetGeneric<T>, NestedIntervalError> {
        for r in intervals {
            if r.start >= r.end {
                return Err(NestedIntervalError::NegativeInterval);
            }
        }
        if intervals.len() != ids.len() {
            return Err(NestedIntervalError::IntervalIdMisMatch);
        }
        let mut idx: Vec<usize> = (0..intervals.len()).collect();
        idx.sort_unstable_by(|idx_a, idx_b| {
            nclist_range_sort(&intervals[*idx_a], &intervals[*idx_b])
        });
        let mut out_iv: Vec<Range<T>> = Vec::with_capacity(intervals.len());
        let mut out_ids: Vec<Vec<u32>> = Vec::with_capacity(intervals.len());
        for ii in 0..idx.len() {
            out_iv.push(intervals[idx[ii]].clone());
            out_ids.push(ids[idx[ii]].clone());
        }
        Ok(IntervalSetGeneric {
            intervals: out_iv,
            ids: out_ids,
            root: RefCell::new(None),
        })
    }

    /// filter this interval set by a bool vec, true are kept
    fn new_filtered(&self, keep: &[bool]) -> IntervalSetGeneric<T> {
        IntervalSetGeneric {
            intervals: self.intervals.filter_by_bools(keep),
            ids: self.ids.filter_by_bools(keep),
            root: RefCell::new(None),
        }
    }

    /// used by the merge functions to bypass the sorting and checking
    /// on already sorted & checked intervals
    fn new_presorted(intervals: Vec<Range<T>>, ids: Vec<Vec<u32>>) -> IntervalSetGeneric<T> {
        IntervalSetGeneric {
            intervals,
            ids,
            root: RefCell::new(None),
        }
    }

    /// How many intervals are there?
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Is the number of intervals 0?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// internal build-the-nested-containment-list-tree
    fn build_tree(
        &self,
        parent: &mut IntervalSetEntry,
        it: &mut std::iter::Peekable<
            std::iter::Enumerate<std::slice::Iter<'_, std::ops::Range<T>>>,
        >,
    ) {
        loop {
            match it.peek() {
                Some((_, next)) => {
                    if (parent.no != -1) && (next.end > self.intervals[parent.no as usize].end) {
                        return;
                    }
                    let (ii, r) = it.next().unwrap();
                    if r.start > r.end {
                        //should be handled by the constructors
                        panic!("invalid interval end < start");
                    }
                    let entry = IntervalSetEntry {
                        no: ii as i32,
                        children: Vec::new(),
                    };
                    parent.children.push(entry);
                    self.build_tree(parent.children.last_mut().unwrap(), it);
                }
                None => {
                    return;
                }
            }
        }
    }

    /// create the nclist if we don't have one yet
    fn ensure_nclist(&self) {
        let mut root = self.root.borrow_mut();
        if root.is_none() {
            let mut new_root = IntervalSetEntry {
                no: -1,
                children: Vec::new(),
            };
            self.build_tree(
                &mut new_root,
                &mut self.intervals.iter().enumerate().peekable(),
            );
            *root = Some(new_root);
        }
    }

    fn depth_first_search<S: IntervalCollector<T>>(
        &self,
        node: &IntervalSetEntry,
        query: &Range<T>,
        collector: &mut S,
    ) {
        let children = &node.children[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop

        ////Todo: is this the correct algorithm from the paper?!
        let first = children
            .upper_bound_by_key(&query.start, |entry| self.intervals[entry.no as usize].end);
        if first == children.len() {
            return;
        }
        for next in &children[first..] {
            let next_iv = &self.intervals[next.no as usize];
            if !next_iv.overlaps(query) {
                return;
            }
            collector.collect(self, next.no as u32);
            if !next.children.is_empty() {
                self.depth_first_search(next, query, collector);
            }
        }
    }
    /// Is there any interval overlapping with the query?
    pub fn has_overlap(&self, query: &Range<T>) -> Result<bool, NestedIntervalError> {
        if query.start > query.end {
            return Err(NestedIntervalError::NegativeInterval);
        }
        self.ensure_nclist();
        //has overlap is easy because all we have to do is scan the first level
        let binding = self.root.borrow();
        let root = binding.as_ref();
        let children = &root.unwrap().children[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first = children
            .upper_bound_by_key(&query.start, |entry| self.intervals[entry.no as usize].end);
        if first == children.len() {
            // ie no entry larger...
            return Ok(false);
        }
        let next = &self.intervals[first];
        Ok(next.overlaps(query))
    }

    /// create an iterator over ```(Range<T>, &vec![id])``` tuples.
    pub fn iter(
        &self,
    ) -> std::iter::Zip<std::slice::Iter<'_, std::ops::Range<T>>, std::slice::Iter<'_, Vec<u32>>>
    {
        self.intervals.iter().zip(self.ids.iter())
    }

    /// retrieve a new IntervalSet with all intervals overlapping the query
    pub fn query_overlapping(&self, query: &Range<T>) -> IntervalSetGeneric<T> {
        self.ensure_nclist();
        let mut collector = VecIntervalCollector::new();
        self.depth_first_search(self.root.borrow().as_ref().unwrap(), query, &mut collector);
        IntervalSetGeneric::new_presorted(collector.intervals, collector.ids)
    }

    /// does this IntervalSet contain overlapping intervals?
    pub fn any_overlapping(&self) -> bool {
        for (next, last) in self.intervals.iter().skip(1).zip(self.intervals.iter()) {
            if last.overlaps(next) {
                return true;
            }
        }
        false
    }
    /// which intervals are overlapping?
    ///
    /// Result is a Vec<bool>
    pub fn overlap_status(&self) -> Vec<bool> {
        let mut result = vec![false; self.intervals.len()];
        for (ii, (next, last)) in self
            .intervals
            .iter()
            .skip(1)
            .zip(self.intervals.iter())
            .enumerate()
        {
            if last.overlaps(next) {
                result[ii] = true; // ii starts at 0
                result[ii + 1] = true;
            }
        }
        result
    }

    /// does this IntervalSet contain nested intervals?
    pub fn any_nested(&self) -> bool {
        self.ensure_nclist();
        for entry in self.root.borrow().as_ref().unwrap().children.iter() {
            if !entry.children.is_empty() {
                return true;
            }
        }
        false
    }

    /// remove intervals that have the same coordinates
    ///
    /// Ids are **not** merged, the first set is being kept
    pub fn remove_duplicates(&self) -> IntervalSetGeneric<T> {
        let mut keep = vec![false; self.len()];
        keep[0] = true;
        for (ix, (v1, v2)) in self
            .intervals
            .iter()
            .skip(1)
            .zip(self.intervals.iter())
            .enumerate()
        {
            keep[ix + 1] = v1 != v2;
        }
        self.new_filtered(&keep)
    }

    /// remove empty intervals, ie. those with start == end
    pub fn remove_empty(&self) -> IntervalSetGeneric<T> {
        let keep: Vec<bool> = self.intervals.iter().map(|r| r.start != r.end).collect();
        self.new_filtered(&keep)
    }

    /// Merge overlapping & nested intervals to their outer bounds
    ///
    /// Examples:
    /// - 0..15, 10..20 -> 0..20
    /// - 0..20, 3..5 -> 0..20
    pub fn merge_hull(&self) -> IntervalSetGeneric<T> {
        let mut new_intervals: Vec<Range<T>> = Vec::new();
        let mut new_ids: Vec<Vec<u32>> = Vec::new();
        let mut it = self.intervals.iter().zip(self.ids.iter()).peekable();
        while let Some(this_element) = it.next() {
            let mut this_iv = this_element.0.start..this_element.0.end;
            let mut this_ids = this_element.1.clone();
            while let Some(next) = it.peek() {
                if next.0.start < this_iv.end {
                    if next.0.end > this_iv.end {
                        this_iv.end = next.0.end;
                    }
                    this_ids.extend_from_slice(next.1);
                    it.next(); // consume that one!
                } else {
                    break;
                }
            }
            new_intervals.push(this_iv);
            this_ids.sort();
            new_ids.push(this_ids)
        }
        IntervalSetGeneric::new_presorted(new_intervals, new_ids)
    }

    /// Merge intervals that are butted up against each other
    ///
    ///This first induces a merge_hull()!
    ///
    /// Examples:
    /// - 0..15, 15..20 -> 0..20
    /// - 0..15, 16..20, 20..30 > 0..15, 16..30
    pub fn merge_connected(&self) -> IntervalSetGeneric<T> {
        let hull = self.merge_hull();
        let mut new_intervals: Vec<Range<T>> = Vec::new();
        let mut new_ids: Vec<Vec<u32>> = Vec::new();
        let mut it = hull.intervals.iter().zip(hull.ids.iter()).peekable();
        while let Some(this_element) = it.next() {
            let mut this_iv = this_element.0.start..this_element.0.end;
            let mut this_ids = this_element.1.clone();
            while let Some(next) = it.peek() {
                if next.0.start == this_iv.end {
                    if next.0.end > this_iv.end {
                        this_iv.end = next.0.end;
                    }
                    this_ids.extend_from_slice(next.1);
                    it.next(); // consume that one!
                } else {
                    break;
                }
            }
            new_intervals.push(this_iv);
            this_ids.sort();
            new_ids.push(this_ids)
        }
        IntervalSetGeneric::new_presorted(new_intervals, new_ids)
    }

    /// Remove all intervals that are overlapping or nested
    /// by simply dropping them.
    ///
    /// Examples:
    /// - 0..20, 10..15, 20..35, 30..40, 40..50 -> 40..50
    ///
    /// Ids of the remaining intervals are unchanged
    pub fn merge_drop(&self) -> IntervalSetGeneric<T> {
        let mut keep = vec![true; self.len()];
        let mut last_stop = T::zero();
        for ii in 0..self.len() {
            if self.intervals[ii].start < last_stop {
                keep[ii] = false;
                keep[ii - 1] = false;
            }
            if self.intervals[ii].end > last_stop {
                last_stop = self.intervals[ii].end;
            }
        }
        self.new_filtered(&keep)
    }

    /// Create fully disjoint intervals by splitting
    /// the existing ones based on their overlap.
    ///
    /// Example:
    /// - 0..20, 10..30 -> 0..10, 10..20, 20..30
    ///
    /// Ids are merged, ie. in the above example,
    /// if the input ids are ```[[0], [1]]```, the output ids are
    /// ```[[0], [0,1], [1]]```
    ///
    /// merge_split on an already disjoint set is a no-op
    pub fn merge_split(&self) -> IntervalSetGeneric<T> {
        #[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug)]
        enum SiteKind {
            End,
            Start,
        }
        #[derive(Debug)]
        struct Site<T> {
            pos: T,
            kind: SiteKind,
            id: Vec<u32>,
        }
        let mut sites: Vec<Site<T>> = Vec::new();
        for (iv, id) in self.intervals.iter().zip(self.ids.iter()) {
            sites.push(Site {
                pos: iv.start,
                kind: SiteKind::Start,
                id: id.clone(),
            });
            sites.push(Site {
                pos: iv.end,
                kind: SiteKind::End,
                id: id.clone(),
            });
        }
        sites.sort_by_key(|x| (x.pos, x.kind));
        let mut new_intervals = Vec::new();
        let mut new_ids: Vec<Vec<u32>> = Vec::new();
        if !sites.is_empty() {
            let mut it = sites.iter_mut();
            let mut last = it.next().unwrap();
            let mut last_ids: HashMap<u32, u32> = HashMap::new();
            for id in &last.id {
                last_ids.insert(*id, 1);
            }
            for next in it {
                //if (last.kind == SiteKind::Start)
                //   || ((last.kind == SiteKind::End) && (next.kind == SiteKind::End))
                {
                    if last.pos != next.pos && !last_ids.is_empty() {
                        new_intervals.push(last.pos..next.pos);
                        let mut ids_here: Vec<u32> = last_ids.keys().cloned().collect();
                        ids_here.sort();
                        new_ids.push(ids_here);
                    }
                }
                match next.kind {
                    SiteKind::Start => {
                        for id in &next.id {
                            *last_ids.entry(*id).or_insert(0) += 1;
                        }
                    }
                    SiteKind::End => {
                        for id in &next.id {
                            *last_ids.get_mut(id).unwrap() -= 1;
                        }
                        last_ids.retain(|_k, v| (*v > 0));
                    }
                }
                last = next;
            }
        }
        IntervalSetGeneric {
            intervals: new_intervals,
            ids: new_ids,
            root: RefCell::new(None),
        }
    }

    /// find the interval with the closest start to the left of pos
    /// None if there are no intervals to the left of pos
    pub fn find_closest_start_left(&self, pos: T) -> Option<(Range<T>, Vec<u32>)> {
        let first = self.intervals.upper_bound_by_key(&pos, |entry| entry.start);
        if first == 0 {
            return None;
        }
        let prev = first - 1;
        Some((self.intervals[prev].clone(), self.ids[prev].clone()))
    }

    /// find the interval with the closest start to the right of pos
    /// None if there are no intervals to the right of pos
    pub fn find_closest_start_right(&self, pos: T) -> Option<(Range<T>, Vec<u32>)> {
        let first = self
            .intervals
            .upper_bound_by_key(&pos, |entry| entry.start + T::one());
        // since this the first element strictly greater, we have to do -1
        if first == self.len() {
            return None;
        }
        Some((self.intervals[first].clone(), self.ids[first].clone()))
    }

    /// find the interval with the closest start to pos
    ///
    /// None if the IntervalSet is empty
    /// On a tie, the left interval wins.
    pub fn find_closest_start(&self, pos: T) -> Option<(Range<T>, Vec<u32>)> {
        let left = self.find_closest_start_left(pos);
        let right = self.find_closest_start_right(pos);
        match (left, right) {
            (None, None) => None,
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (Some(l), Some(r)) => {
                /* let distance_left = (i64::from(l.0.start) - i64::from(pos)).abs();
                let distance_right = (i64::from(r.0.start) - i64::from(pos)).abs(); */
                let distance_left = if l.0.start > pos {
                    l.0.start - pos
                } else {
                    pos - l.0.start
                };
                let distance_right = if r.0.start > pos {
                    r.0.start - pos
                } else {
                    pos - r.0.start
                };

                if distance_left <= distance_right {
                    Some(l)
                } else {
                    Some(r)
                }
            }
        }
    }

    /// how many units in interval space does this IntervalSet cover
    pub fn covered_units(&self) -> T {
        let merged = self.merge_hull();
        let mut total = T::zero();
        for iv in merged.intervals.iter() {
            total = total + iv.end - iv.start;
        }
        total
    }

    /// What is the mean size of the intervals
    #[allow(clippy::cast_lossless)]
    pub fn mean_interval_size(&self) -> f64 {
        let mut total = T::zero();
        for iv in self.intervals.iter() {
            total = total + iv.end - iv.start;
        }
        let total: f64 = total.to_f64().unwrap();
        total / self.len() as f64
    }

    /// Invert the intervals in this set
    ///
    /// Actual applied lower_bound is min(lower_bound, first_interval_start)
    /// Actual applied upper_bound is max(upper_bound, last_interval_end)
    ///
    /// Examples
    /// - invert([15..20], 0, 30) -> [0..15, 20..30]
    /// - invert([15..20], 20, 30) -> [0..15, 20..30]
    ///
    /// Ids are lost
    pub fn invert(&self, lower_bound: T, upper_bound: T) -> IntervalSetGeneric<T> {
        let mut new_intervals: Vec<Range<T>> = Vec::new();
        let mut new_ids: Vec<Vec<u32>> = Vec::new();

        if self.is_empty() {
            new_intervals.push(lower_bound..upper_bound);
            new_ids.push(vec![0]);
        } else {
            let lower = min(lower_bound, self.intervals[0].start);
            let upper = max(upper_bound, self._highest_end().unwrap());
            let n = self.merge_hull();
            let mut paired = vec![lower];
            for iv in n.intervals {
                paired.push(iv.start);
                paired.push(iv.end);
            }
            paired.push(upper);
            new_intervals.extend(
                paired
                    .chunks(2)
                    .filter(|se| se[0] != se[1])
                    .map(|se| se[0]..se[1]),
            );
            new_ids.extend((0..new_intervals.len()).map(|x| vec![x as u32]));
        }
        IntervalSetGeneric::new_presorted(new_intervals, new_ids)
    }

    /// Filter to those intervals that have an overlap in other.
    pub fn filter_to_overlapping(
        &self,
        other: &mut IntervalSetGeneric<T>,
    ) -> IntervalSetGeneric<T> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        self.ensure_nclist();
        let mut collector = TagIntervalCollector::new(self);
        for q in other.intervals.iter() {
            self.depth_first_search(self.root.borrow().as_ref().unwrap(), q, &mut collector);
        }
        self.new_filtered(&collector.hit)
    }

    /// Filter to those intervals having an overlap in other, and split/truncate them
    /// so the resulting intervals are fully contained within the intervals covered by other.
    /// ids are being kept. 
    ///
    /// so 10..20 vs 5..11, 15..17, 18..30 -> 10..11, 15..17, 18..20
    pub fn filter_to_overlapping_and_split(
        &self,
        other: &IntervalSetGeneric<T>,
    ) -> IntervalSetGeneric<T> {
        let mut out_ivs = Vec::new();
        let mut out_ids = Vec::new();

        let other = other.merge_connected();
        for (ii, iv) in self.intervals.iter().enumerate() {
            let id = &self.ids[ii];
            let overlapping = other.query_overlapping(iv).merge_connected();
            for (new_iv, _) in overlapping.iter() {
                //remember, these are disjoint
                let start = max(iv.start, new_iv.start);
                let stop = min(iv.end, new_iv.end);
                out_ivs.push(start..stop);
                out_ids.push(id.clone());
            }
        }

        IntervalSetGeneric::new_with_ids_multiple(&out_ivs, out_ids)
            .expect("Unexpected failure in rebuilding intervals")
    }

    /// Filter to those intervals that have no overlap in other.
    pub fn filter_to_non_overlapping(
        &self,
        other: &mut IntervalSetGeneric<T>,
    ) -> IntervalSetGeneric<T> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here

        let keep: Result<Vec<bool>, NestedIntervalError> = self
            .intervals
            .iter()
            .enumerate()
            .map(|(_ii, iv)| other.has_overlap(iv))
            .map_ok(|x| !x)
            .collect();
        match keep {
            Ok(keep) => self.new_filtered(&keep),
            Err(_) => panic!(
                "Negative intervals encountered inside IntervalSets - check input sanity code"
            ),
        }
    }

    /// Filter to those intervals that have an overlap in at least k others.
    pub fn filter_to_overlapping_k_others(
        &self,
        others: &[&IntervalSetGeneric<T>],
        min_k: u32,
    ) -> IntervalSetGeneric<T> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)

        let counts = self._count_overlapping(others);
        let mut keep = vec![false; self.len()];
        for (ii, value) in counts.iter().enumerate() {
            if *value >= min_k {
                keep[ii] = true;
            }
        }
        self.new_filtered(&keep)
    }
    /// Filter to those intervals that have an overlap in no more than k others
    pub fn filter_to_non_overlapping_k_others(
        &self,
        others: &[&IntervalSetGeneric<T>],
        max_k: u32,
    ) -> IntervalSetGeneric<T> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        let counts = self._count_overlapping(others);
        let mut keep = vec![false; self.len()];
        for (ii, value) in counts.iter().enumerate() {
            if *value <= max_k {
                keep[ii] = true;
            }
        }
        self.new_filtered(&keep)
    }

    /// Build the union of two IntervalSets
    ///
    /// No merging is performed
    pub fn union(&self, others: &[&IntervalSetGeneric<T>]) -> IntervalSetGeneric<T> {
        let mut new_intervals: Vec<Range<T>> = Vec::new();
        new_intervals.extend_from_slice(&self.intervals);
        for o in others.iter() {
            new_intervals.extend_from_slice(&o.intervals);
        }
        IntervalSetGeneric::new(&new_intervals[..]).unwrap()
    }

    /// find the highest interval.end in our data set
    fn _highest_end(&self) -> Option<T> {
        match self.root.borrow().as_ref() {
            // if we have a nclist we can just look at those intervals.
            Some(root) => root
                .children
                .iter()
                .map(|entry| self.intervals[entry.no as usize].end)
                .max(),
            //and if we don't we have to brutforce, I believe, since our list is sorted by start
            None => self.intervals.iter().map(|range| range.end).max(),
        }
    }

    fn _count_overlapping(&self, others: &[&IntervalSetGeneric<T>]) -> Vec<u32> {
        self.ensure_nclist();
        let mut counts: Vec<u32> = vec![0; self.len()];
        for o in others {
            let mut collector = TagIntervalCollector::new(self);
            for q in o.intervals.iter() {
                self.depth_first_search(self.root.borrow().as_ref().unwrap(), q, &mut collector);
            }
            for (ii, value) in collector.hit.iter().enumerate() {
                if *value {
                    counts[ii] += 1;
                }
            }
        }
        counts
    }
}

impl<T: Rangable + std::fmt::Debug> Eq for IntervalSetGeneric<T> {}
impl<T: Rangable + std::fmt::Debug> PartialEq for IntervalSetGeneric<T> {
    fn eq(&self, other: &IntervalSetGeneric<T>) -> bool {
        (self.intervals == other.intervals) && (self.ids == other.ids)
    }
}

/// Extend Range functionality with some often used bool functions
pub trait RangePlus<T> {
    /// Does this interval overlap the other one?
    fn overlaps(&self, other: &Range<T>) -> bool;
}

impl<T: Rangable> RangePlus<T> for Range<T> {
    fn overlaps(&self, other: &Range<T>) -> bool {
        self.start < other.end && other.start < self.end && other.start < other.end
    }
}

#[cfg(test)]
#[allow(dead_code)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use crate::{IntervalSet, IntervalSetGeneric};
    use std::ops::Range;
    #[test]
    fn test_has_overlap() {
        let r = vec![0..5, 10..15];
        let n = IntervalSet::new(&r).unwrap();
        assert!(n.has_overlap(&(3..4)).unwrap());
        assert!(n.has_overlap(&(5..20)).unwrap());
        assert!(!n.has_overlap(&(6..10)).unwrap());
        assert!(!n.has_overlap(&(100..110)).unwrap());
        assert!(!n.has_overlap(&(3..3)).unwrap());

        let r2 = vec![0..15, 0..6];
        let n = IntervalSet::new(&r2).unwrap();
        assert!(n.has_overlap(&(3..4)).unwrap());
        assert!(n.has_overlap(&(5..20)).unwrap());
        assert!(n.has_overlap(&(6..10)).unwrap());
        assert!(!n.has_overlap(&(20..30)).unwrap());

        let r2 = vec![100..150, 30..40, 200..400];
        let n = IntervalSet::new(&r2).unwrap();
        assert!(n.has_overlap(&(101..102)).unwrap());
        assert!(n.has_overlap(&(149..150)).unwrap());
        assert!(n.has_overlap(&(39..99)).unwrap());
        assert!(n.has_overlap(&(29..99)).unwrap());
        assert!(n.has_overlap(&(19..99)).unwrap());
        assert!(!n.has_overlap(&(0..5)).unwrap());
        assert!(!n.has_overlap(&(0..29)).unwrap());
        assert!(!n.has_overlap(&(0..30)).unwrap());
        assert!(n.has_overlap(&(0..31)).unwrap());
        assert!(!n.has_overlap(&(40..41)).unwrap());
        assert!(!n.has_overlap(&(40..99)).unwrap());
        assert!(!n.has_overlap(&(40..100)).unwrap());
        assert!(n.has_overlap(&(40..101)).unwrap());
        assert!(n.has_overlap(&(399..400)).unwrap());
        assert!(!n.has_overlap(&(400..4000)).unwrap());
    }

    #[test]
    fn test_iter() {
        let n = IntervalSet::new(&[100..150, 30..40, 200..400, 250..300]).unwrap();
        let c: Vec<(&Range<u32>, &Vec<u32>)> = n.iter().collect();
        assert!(!c.is_empty());
        let c: Vec<Range<u32>> = c
            .iter()
            .map(|(interval, _id)| (*interval).clone())
            .collect();
        dbg!(&c);
        assert!(c == vec![30..40, 100..150, 200..400, 250..300]);
    }

    #[test]
    fn test_query() {
        let n = IntervalSet::new(&[100..150, 30..40, 200..400, 250..300]).unwrap();
        let c = n.query_overlapping(&(0..5));
        assert!(c.is_empty());
        let c = n.query_overlapping(&(0..31));
        assert_eq!(c.intervals, vec![30..40]);
        let c = n.query_overlapping(&(200..250));
        assert_eq!(c.intervals, vec![200..400]);
        let c = n.query_overlapping(&(200..251));
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c = n.query_overlapping(&(0..1000));
        dbg!(&c);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.query_overlapping(&(401..1000));
        assert!(c.is_empty());
    }
    #[test]
    fn test_query_multiple() {
        let n = IntervalSet::new(&[100..150, 30..40, 200..400, 250..300]).unwrap();
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&[0..5, 0..105]).unwrap());
        assert_eq!(c.intervals, vec![30..40, 100..150]);
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&[500..600, 550..700]).unwrap());
        assert!(c.is_empty());
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&[45..230]).unwrap());
        assert_eq!(c.intervals, vec![100..150, 200..400]);
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&[45..101, 101..230]).unwrap());
        assert_eq!(c.intervals, vec![100..150, 200..400]);
    }

    #[test]
    fn test_query_multiple_non_overlapping() {
        let n = IntervalSet::new(&[100..150, 30..40, 200..400, 250..300]).unwrap();
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&[0..5, 0..105]).unwrap());
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&[500..600, 550..700]).unwrap());
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&[0..600]).unwrap());
        assert!(c.is_empty());
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&[45..230]).unwrap());
        assert_eq!(c.intervals, vec![30..40, 250..300]);
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&[45..101, 101..230]).unwrap());
        assert_eq!(c.intervals, vec![30..40, 250..300]);
    }

    #[test]
    fn test_any_overlapping() {
        let n = IntervalSet::new(&[100..150]).unwrap();
        assert!(!n.any_overlapping());
        let n = IntervalSet::new(&[100..150, 200..300]).unwrap();
        assert!(!n.any_overlapping());
        let n = IntervalSet::new(&[100..150, 150..300]).unwrap();
        assert!(!n.any_overlapping());
        let n = IntervalSet::new(&[100..151, 150..300]).unwrap();
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&[100..151, 105..110]).unwrap();
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&[100..151, 105..110, 0..1000]).unwrap();
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&[100..150, 150..210, 0..1000]).unwrap();
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&[100..150, 150..210, 0..130]).unwrap();
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&[100..150, 150..210, 150..250]).unwrap();
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&[100..150, 150..210, 149..250]).unwrap();
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&[100..150, 150..210, 209..250]).unwrap();
        assert!(n.any_overlapping());
    }

    #[test]
    fn test_any_nested() {
        assert!(!IntervalSet::new(&[]).unwrap().any_nested());
        assert!(!IntervalSet::new(&[100..150]).unwrap().any_nested());
        assert!(!IntervalSet::new(&[100..150, 150..300])
            .unwrap()
            .any_nested());
        assert!(!IntervalSet::new(&[100..151, 150..300])
            .unwrap()
            .any_nested());
        assert!(IntervalSet::new(&[100..151, 150..300, 100..130])
            .unwrap()
            .any_nested());
        assert!(IntervalSet::new(&[100..151, 150..300, 0..1000])
            .unwrap()
            .any_nested());
    }

    #[test]
    fn test_remove_duplicates() {
        let n = IntervalSet::new(&[100..150]).unwrap().remove_duplicates();
        assert!(!n.any_overlapping());
        assert_eq!(n.len(), 1);

        let n = IntervalSet::new(&[30..40, 30..40, 100..150])
            .unwrap()
            .remove_duplicates();
        assert!(!n.any_overlapping());
        assert_eq!(n.len(), 2);
        let n = IntervalSet::new(&[30..40, 30..40, 35..150])
            .unwrap()
            .remove_duplicates();
        assert_eq!(n.len(), 2);
        let n = IntervalSet::new_with_ids(
            &[30..40, 30..40, 35..150, 35..150, 36..38],
            &[55, 56, 57, 58, 59],
        )
        .unwrap()
        .remove_duplicates();
        assert_eq!(n.len(), 3);
        dbg!(&n);
        assert_eq!(n.ids, vec![vec![55], vec![57], vec![59]]);
    }

    #[test]
    fn test_merge_hull() {
        let n = IntervalSet::new(&[100..150, 120..180, 110..115, 200..201])
            .unwrap()
            .merge_hull();
        assert_eq!(n.intervals, vec![100..180, 200..201]);
        assert_eq!(n.ids, vec![vec![0, 1, 2], vec![3]]);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&[100..150]).unwrap().merge_hull();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&[100..150, 120..180])
            .unwrap()
            .merge_hull();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![100..180]);
        assert_eq!(n.ids, vec![vec![0, 1]]);

        let n = IntervalSet::new(&[100..150, 120..180, 110..115])
            .unwrap()
            .merge_hull();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![100..180]);
        assert_eq!(n.ids, vec![vec![0, 1, 2]]);

        let n =
            IntervalSet::new_with_ids(&[300..400, 400..450, 450..500, 510..520], &[20, 10, 30, 40])
                .unwrap();
        assert_eq!(n.intervals, vec![300..400, 400..450, 450..500, 510..520]);
    }

    #[test]
    fn test_merge_drop() {
        let n = IntervalSet::new(&[]).unwrap().merge_drop();
        assert_eq!(n.len(), 0);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&[100..150]).unwrap().merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&[100..150, 120..180])
            .unwrap()
            .merge_drop();
        assert_eq!(n.len(), 0);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![]);
        assert_eq!(n.ids, Vec::<Vec<u32>>::new());

        let n = IntervalSet::new(&[100..150, 120..180, 200..250])
            .unwrap()
            .merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![2]]);

        let n = IntervalSet::new(&[100..150, 120..180, 200..250, 106..110])
            .unwrap()
            .merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![3]]);

        let n = IntervalSet::new(&[100..150, 120..180, 200..250, 106..110, 80..105])
            .unwrap()
            .merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![4]]);

        let n = IntervalSet::new(&[100..150, 120..180, 200..250, 106..110, 80..105, 30..40])
            .unwrap()
            .merge_drop();
        assert_eq!(n.len(), 2);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![30..40, 200..250]);
        assert_eq!(n.ids, vec![vec![0], vec![5]]);

        let n = IntervalSet::new(&[
            100..150,
            120..180,
            200..250,
            106..110,
            80..105,
            30..40,
            400..405,
        ])
        .unwrap()
        .merge_drop();
        assert_eq!(n.len(), 3);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![30..40, 200..250, 400..405]);
        assert_eq!(n.ids, vec![vec![0], vec![5], vec![6]]);
        let n = IntervalSet::new(&[0..20, 10..15, 20..35, 30..40, 40..50])
            .unwrap()
            .merge_drop();
        assert_eq!(n.intervals, vec![40..50]);
    }

    #[test]
    fn test_find_closest_start_left() {
        let n = IntervalSet::new(&[
            30..40,
            80..105,
            100..150,
            106..110,
            106..120,
            107..125,
            120..180,
            200..250,
            400..405,
        ])
        .unwrap();
        //find the first range that has an end to the left of this
        assert!(n.find_closest_start_left(29).is_none());
        assert_eq!(n.find_closest_start_left(100).unwrap(), (100..150, vec![2]));
        assert_eq!(n.find_closest_start_left(105).unwrap(), (100..150, vec![2]));
        assert_eq!(n.find_closest_start_left(106).unwrap(), (106..110, vec![4]));
        assert_eq!(n.find_closest_start_left(109).unwrap(), (107..125, vec![5]));
        assert_eq!(n.find_closest_start_left(110).unwrap(), (107..125, vec![5]));
        assert_eq!(n.find_closest_start_left(111).unwrap(), (107..125, vec![5]));
        assert_eq!(n.find_closest_start_left(120).unwrap(), (120..180, vec![6]));
        assert_eq!(n.find_closest_start_left(121).unwrap(), (120..180, vec![6]));
        assert_eq!(n.find_closest_start_left(125).unwrap(), (120..180, vec![6]));
        assert_eq!(n.find_closest_start_left(127).unwrap(), (120..180, vec![6]));
        assert_eq!(
            n.find_closest_start_left(121000).unwrap(),
            (400..405, vec![8])
        );
        let n = IntervalSet::new(&[]).unwrap();
        assert!(n.find_closest_start_left(29).is_none());
    }

    #[test]
    fn test_find_closest_start_right() {
        let n = IntervalSet::new(&[
            30..40,
            80..105,
            100..150,
            106..110,
            106..120,
            107..125,
            120..180,
            200..250,
            400..405,
        ])
        .unwrap();
        //find the first range that has an end to the right of this
        assert_eq!(n.find_closest_start_right(10).unwrap(), (30..40, vec![0]));
        assert_eq!(n.find_closest_start_right(29).unwrap(), (30..40, vec![0]));
        assert_eq!(n.find_closest_start_right(30).unwrap(), (30..40, vec![0]));
        assert_eq!(n.find_closest_start_right(31).unwrap(), (80..105, vec![1]));
        assert_eq!(n.find_closest_start_right(99).unwrap(), (100..150, vec![2]));
        assert_eq!(
            n.find_closest_start_right(100).unwrap(),
            (100..150, vec![2])
        );
        assert_eq!(
            n.find_closest_start_right(101).unwrap(),
            (106..120, vec![3])
        );
        assert_eq!(
            n.find_closest_start_right(107).unwrap(),
            (107..125, vec![5])
        );
        assert_eq!(
            n.find_closest_start_right(110).unwrap(),
            (120..180, vec![6])
        );
        assert_eq!(
            n.find_closest_start_right(111).unwrap(),
            (120..180, vec![6])
        );
        assert_eq!(
            n.find_closest_start_right(120).unwrap(),
            (120..180, vec![6])
        );
        assert_eq!(
            n.find_closest_start_right(121).unwrap(),
            (200..250, vec![7])
        );
        assert_eq!(
            n.find_closest_start_right(125).unwrap(),
            (200..250, vec![7])
        );
        assert_eq!(
            n.find_closest_start_right(127).unwrap(),
            (200..250, vec![7])
        );
        assert!(n.find_closest_start_right(121000).is_none());
        let n = IntervalSet::new(&[]).unwrap();
        assert!(n.find_closest_start_right(29).is_none());
    }

    #[test]
    fn test_find_closest_start() {
        let n = IntervalSet::new(&[]).unwrap();
        assert!(n.find_closest_start(100).is_none());
        let n = IntervalSet::new(&[100..110, 200..300]).unwrap();
        assert_eq!(n.find_closest_start(0).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(100).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(149).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(150).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(151).unwrap(), (200..300, vec![1]));
        assert_eq!(n.find_closest_start(251).unwrap(), (200..300, vec![1]));
        assert_eq!(n.find_closest_start(351).unwrap(), (200..300, vec![1]));

        let n = IntervalSet::new(&[10..11, 1000..1110]).unwrap();
        assert_eq!(n.find_closest_start(5).unwrap(), (10..11, vec![0]));
        let n = IntervalSet::new(&[
            566564..667063,
            569592..570304,
            713866..714288,
            935162..937142,
            1051311..1052403,
            1279151..1281233,
            1282803..1283631,
            1310387..1311060,
            1337193..1337881,
            1447089..1447626,
        ])
        .unwrap();

        assert_eq!(
            n.find_closest_start(570000).unwrap(),
            (569592..570304, vec![1])
        );
    }

    #[test]
    fn test_covered_units() {
        let n = IntervalSet::new(&[]).unwrap();
        assert_eq!(n.covered_units(), 0);
        let n = IntervalSet::new(&[10..100]).unwrap();
        assert_eq!(n.covered_units(), 90);
        let n = IntervalSet::new(&[10..100, 200..300]).unwrap();
        assert_eq!(n.covered_units(), 90 + 100);
        let n = IntervalSet::new(&[10..100, 200..300, 15..99]).unwrap();
        assert_eq!(n.covered_units(), 90 + 100);
        let n = IntervalSet::new(&[10..100, 200..300, 15..99, 15..105]).unwrap();
        assert_eq!(n.covered_units(), 90 + 100 + 5);
    }

    #[test]
    fn test_mean_interval_size() {
        let n = IntervalSet::new(&[]).unwrap();
        assert!(n.mean_interval_size().is_nan());
        let n = IntervalSet::new(&[10..100]).unwrap();
        assert_eq!(n.mean_interval_size(), 90.);
        let n = IntervalSet::new(&[10..100, 200..300]).unwrap();
        assert_eq!(n.mean_interval_size(), (90 + 100) as f64 / 2.0);
        let n = IntervalSet::new(&[10..100, 200..300, 15..99]).unwrap();
        assert_eq!(n.mean_interval_size(), (90 + 100 + (99 - 15)) as f64 / 3.0);
        let n = IntervalSet::new(&[10..100, 200..300, 15..99, 15..105]).unwrap();
        assert_eq!(
            n.mean_interval_size(),
            (((100 - 10) + (300 - 200) + (99 - 15) + (105 - 15)) as f64 / 4.0)
        );
    }

    #[test]
    fn test_invert() {
        let n = IntervalSet::new(&[]).unwrap().invert(0, 100);
        assert_eq!(n.intervals, vec![0..100,]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = IntervalSet::new(&[30..40]).unwrap().invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 40..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = IntervalSet::new(&[30..40, 35..38]).unwrap().invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 40..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = IntervalSet::new(&[30..40, 35..38, 35..50])
            .unwrap()
            .invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 50..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = IntervalSet::new(&[30..40, 35..38, 35..50])
            .unwrap()
            .invert(40, 100);
        assert_eq!(n.intervals, vec![50..100,]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = IntervalSet::new(&[30..40, 35..38, 35..50, 55..60])
            .unwrap()
            .invert(40, 40);
        assert_eq!(n.intervals, vec![50..55]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = IntervalSet::new(&[30..40, 35..38, 35..50])
            .unwrap()
            .invert(40, 40);

        assert!(n.intervals.is_empty());
        assert!(n.ids.is_empty());

        // this of course only works for distinct intervals
        let n = IntervalSet::new(&[10..20, 35..38, 40..50]).unwrap();
        let ni = n.invert(0, 50);
        let nii = ni.invert(0, 50);
        assert_eq!(n.intervals, nii.intervals);

        let n = IntervalSet::new(&[10..36, 35..38, 40..50]).unwrap();
        let ni = n.invert(0, 100);
        let n2 = n.union(&[&ni]).merge_connected();
        assert_eq!(n2.intervals, vec![0..100]);

        let n = IntervalSet::new(&[]).unwrap();
        let ni = n.invert(0, 100);
        assert_eq!(ni.intervals, vec![0..100]);
    }

    #[test]
    fn test_union() {
        let n = IntervalSet::new(&[])
            .unwrap()
            .union(&[&IntervalSet::new(&[0..100]).unwrap()]);
        assert_eq!(n.intervals, vec![0..100]);

        let n = IntervalSet::new(&[0..10])
            .unwrap()
            .union(&[&IntervalSet::new(&[0..100]).unwrap()]);
        assert_eq!(n.intervals, vec![0..100, 0..10]);

        let n = IntervalSet::new(&[0..10])
            .unwrap()
            .union(&[&IntervalSet::new(&[0..100, 200..300]).unwrap()]);
        assert_eq!(n.intervals, vec![0..100, 0..10, 200..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![2]]);

        let n = IntervalSet::new(&[0..10])
            .unwrap()
            .union(&[&IntervalSet::new(&[]).unwrap()]);
        assert_eq!(n.intervals, vec![0..10]);
        let n = IntervalSet::new(&[0..10]).unwrap().union(&[
            &IntervalSet::new(&[0..100]).unwrap(),
            &IntervalSet::new(&[200..300]).unwrap(),
        ]);
        assert_eq!(n.intervals, vec![0..100, 0..10, 200..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn test_substract() {
        let n = IntervalSet::new(&[])
            .unwrap()
            .filter_to_non_overlapping(&mut IntervalSet::new(&[0..100]).unwrap());
        assert!(n.intervals.is_empty());

        let n = IntervalSet::new(&[0..10])
            .unwrap()
            .filter_to_non_overlapping(&mut IntervalSet::new(&[0..100]).unwrap());
        assert!(n.intervals.is_empty());

        let n = IntervalSet::new(&[0..10, 100..150])
            .unwrap()
            .filter_to_non_overlapping(&mut IntervalSet::new(&[0..100]).unwrap());
        assert_eq!(n.intervals, vec![100..150]);

        let n = IntervalSet::new(&[0..10, 100..150, 150..300])
            .unwrap()
            .filter_to_non_overlapping(&mut IntervalSet::new(&[55..101]).unwrap());
        assert_eq!(n.intervals, vec![0..10, 150..300]);
        assert_eq!(n.ids, vec![vec![0], vec![2]]);

        let n = IntervalSet::new(&[0..10, 5..6, 100..150, 150..300])
            .unwrap()
            .filter_to_non_overlapping(&mut IntervalSet::new(&[55..101]).unwrap());
        assert_eq!(n.intervals, vec![0..10, 5..6, 150..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![3]]);
    }

    #[test]
    fn test_filter_overlapping_multiples() {
        let n = IntervalSet::new(&[100..150, 30..40, 200..400, 250..300]).unwrap();
        let c = n.filter_to_overlapping_k_others(&[&IntervalSet::new(&[0..5, 0..105]).unwrap()], 1);
        assert_eq!(c.intervals, vec![30..40, 100..150]);
        let c = n.filter_to_overlapping_k_others(&[&IntervalSet::new(&[0..5, 0..105]).unwrap()], 0);
        assert_eq!(c, n);
        let c = n.filter_to_overlapping_k_others(&[&IntervalSet::new(&[0..5, 0..105]).unwrap()], 2);
        assert!(c.is_empty());

        let c = n.filter_to_overlapping_k_others(
            &[
                &IntervalSet::new(&[0..35]).unwrap(),
                &IntervalSet::new(&[0..160]).unwrap(),
            ],
            2,
        );
        assert_eq!(c.intervals, vec![30..40,]);
        let c = n.filter_to_overlapping_k_others(
            &[
                &IntervalSet::new(&[0..35]).unwrap(),
                &IntervalSet::new(&[0..160]).unwrap(),
            ],
            1,
        );
        assert_eq!(c.intervals, vec![30..40, 100..150]);
    }

    #[test]
    fn test_filter_non_overlapping_multiples() {
        let n = IntervalSet::new(&[100..150, 30..40, 200..400, 250..300]).unwrap();
        let c =
            n.filter_to_non_overlapping_k_others(&[&IntervalSet::new(&[0..5, 0..105]).unwrap()], 1);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c =
            n.filter_to_non_overlapping_k_others(&[&IntervalSet::new(&[0..5, 0..105]).unwrap()], 0);
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c =
            n.filter_to_non_overlapping_k_others(&[&IntervalSet::new(&[0..5, 0..105]).unwrap()], 2);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);

        let c = n.filter_to_non_overlapping_k_others(
            &[
                &IntervalSet::new(&[0..35]).unwrap(),
                &IntervalSet::new(&[0..160]).unwrap(),
            ],
            2,
        );
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(
            &[
                &IntervalSet::new(&[0..35]).unwrap(),
                &IntervalSet::new(&[0..160]).unwrap(),
            ],
            1,
        );
        assert_eq!(c.intervals, vec![100..150, 200..400, 250..300]);
    }

    #[test]
    fn test_split() {
        let n = IntervalSet::new(&[0..100, 20..30]).unwrap();
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..20, 20..30, 30..100]);
        assert_eq!(c.ids, vec![vec![0], vec![0, 1,], vec![0]]);

        let n = IntervalSet::new(&[0..100, 0..90, 70..95, 110..150]).unwrap();
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..70, 70..90, 90..95, 95..100, 110..150]);
        assert_eq!(
            c.ids,
            vec![vec![0, 1], vec![0, 1, 2], vec![0, 2], vec![0], vec![3]]
        );
        let n =
            IntervalSet::new_with_ids(&[0..100, 0..90, 70..95, 110..150], &[100, 200, 300, 400])
                .unwrap();
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..70, 70..90, 90..95, 95..100, 110..150]);
        assert_eq!(
            c.ids,
            vec![
                vec![100, 200],
                vec![100, 200, 300],
                vec![100, 300],
                vec![100],
                vec![400]
            ]
        );
        let d = c.merge_split();
        assert_eq!(c, d);

        let n = IntervalSet::new_with_ids(&[0..100, 5..10, 15..20], &[100, 200, 300]).unwrap();
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..5, 5..10, 10..15, 15..20, 20..100]);
        assert_eq!(
            c.ids,
            vec![
                vec![100],
                vec![100, 200],
                vec![100],
                vec![100, 300],
                vec![100]
            ]
        );

        let n = IntervalSet::new_with_ids(&[0..100, 5..50, 10..15, 25..75], &[100, 200, 300, 400])
            .unwrap();
        let c = n.merge_split();
        assert_eq!(
            c.intervals,
            [0..5, 5..10, 10..15, 15..25, 25..50, 50..75, 75..100]
        );
        assert_eq!(
            c.ids,
            vec![
                vec![100],           // 0..5
                vec![100, 200],      // 5..10
                vec![100, 200, 300], //10..15
                vec![100, 200],      //15..25
                vec![100, 200, 400], //25..50
                vec![100, 400],      //50..75
                vec![100]            //75..100
            ]
        );
    }

    #[test]
    fn test_example() {
        let intervals = vec![0..20, 15..30, 50..100];
        let interval_set = IntervalSet::new(&intervals).unwrap();
        assert_eq!(interval_set.ids, vec![vec![0], vec![1], vec![2]]); // automatic ids, use new_with_ids otherwise
        let hits = interval_set.query_overlapping(&(10..16));
        assert_eq!(hits.intervals, [0..20, 15..30]);
        let merged = hits.merge_hull();
        assert_eq!(merged.intervals, [0..30]);
        assert_eq!(merged.ids, vec![vec![0, 1]]);
    }

    #[test]
    fn test_new_with_ids_sorting() {
        let n = IntervalSet::new_with_ids(&[300..400, 30..40], &[20, 10]).unwrap();
        assert_eq!(n.intervals, [30..40, 300..400]);
        assert_eq!(n.ids, vec![vec![10], vec![20]]);
    }

    #[test]
    fn test_merge_connected() {
        let n = IntervalSet::new_with_ids(&[300..400, 400..450, 450..500], &[20, 10, 30]).unwrap();
        assert_eq!(n.merge_hull().intervals, vec![300..400, 400..450, 450..500]);
        let n = n.merge_connected();
        assert_eq!(n.intervals, [300..500]);
        assert_eq!(n.ids, vec![vec![10, 20, 30],]);

        let n = IntervalSet::new_with_ids(&[300..400, 400..450, 451..500], &[20, 10, 30])
            .unwrap()
            .merge_connected();
        assert_eq!(n.intervals, [300..450, 451..500]);
        assert_eq!(n.ids, vec![vec![10, 20], vec![30]]);
        let n =
            IntervalSet::new_with_ids(&[300..400, 400..450, 451..500, 350..355], &[20, 10, 30, 40])
                .unwrap()
                .merge_connected();
        assert_eq!(n.intervals, [300..450, 451..500]);
        assert_eq!(n.ids, vec![vec![10, 20, 40], vec![30]]);
        let n =
            IntervalSet::new_with_ids(&[300..400, 400..450, 450..500, 510..520], &[20, 10, 30, 40])
                .unwrap()
                .merge_connected();
        assert_eq!(n.intervals, vec![300..500, 510..520]);
    }

    #[test]
    fn test_clone() {
        let n = IntervalSet::new_with_ids(&[300..400, 400..450, 450..500], &[20, 10, 30]).unwrap();
        let n2 = n.clone();
        assert_eq!(n.intervals, n2.intervals);
        assert_eq!(n.ids, n2.ids);
        assert!(n2.root.borrow().is_none());
        n.has_overlap(&(0..1)).unwrap();
        assert!(n2.root.borrow().is_none());
        let n2 = n.clone();
        assert!(n2.root.borrow().is_some());
    }

    #[test]
    fn test_split_repeated_ids() {
        let intervals = vec![
            10736170..10736283,
            10939387..10939423,
            10940596..10940707,
            10941690..10941780,
            10944966..10945053,
            10947303..10947418,
            10949211..10949269,
            10950048..10950174,
            10959066..10959136,
            10961282..10961338,
            10939423..10940596,
            10940707..10941690,
            10941780..10944966,
            10945053..10947303,
            10947418..10949211,
            10949269..10950048,
            10950174..10959066,
            10959136..10961282,
            11066417..11066515,
            11067984..11068174,
            11066515..11067984,
            11124336..11124379,
            11124507..11125705,
            11124379..11124507,
            11249808..11249959,
            12602465..12602527,
            12604669..12604739,
            12615408..12615534,
            12616313..12616371,
            12618170..12618289,
            12620837..12620938,
            12624241..12624331,
            12625316..12625428,
            12626606..12626642,
            12602527..12604669,
            12604739..12615408,
            12615534..12616313,
            12616371..12618170,
            12618289..12620837,
            12620938..12624241,
            12624331..12625316,
            12625428..12626606,
            15273854..15273961,
            15282556..15288670,
            15290717..15290836,
            15290994..15291024,
            15295331..15295410,
            15295729..15295832,
            15297156..15297196,
            15290836..15290994,
            15291024..15295331,
            15295410..15295729,
            15295832..15297156,
            15298377..15304556,
            15326036..15327756,
            15342359..15342426,
            15342944..15343065,
            15327756..15342359,
            15342426..15342944,
            15349466..15350043,
            15489989..15490131,
            15490871..15490947,
            15492485..15492564,
            15490131..15490871,
            15490947..15492485,
            15489989..15490131,
            15490871..15490947,
            15492485..15492564,
            15541435..15541607,
            15558445..15558626,
            15575611..15575786,
            15577966..15578006,
            15541607..15558445,
            15558626..15575611,
            15575786..15577966,
            15550903..15552887,
            15553210..15553586,
            15553690..15553838,
            15552887..15553210,
            15553586..15553690,
            15557576..15558591,
            15560155..15560264,
            15560524..15560694,
            15558591..15560155,
            15560264..15560524,
            15562016..15564276,
            15572088..15573265,
            15588360..15588478,
            15600907..15602514,
            15604051..15604133,
            15604841..15604882,
            15602514..15604051,
            15604133..15604841,
            15611758..15613096,
            15615401..15615578,
            15622600..15622712,
            15623986..15624071,
            15624538..15624674,
            15624955..15625094,
        ];
        let ids = vec![
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 9, 5, 5, 9, 6, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 5, 6, 5, 5, 5, 5, 5, 9, 9, 9, 9, 6, 6,
            6, 6, 10, 10, 5, 5, 5, 5, 9, 9, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 6, 6, 6, 10, 10, 6, 6, 6,
            10, 10, 6, 6, 5, 6, 6, 6, 10, 10, 6, 6, 6, 6, 6, 6,
        ];

        let n = IntervalSet::new_with_ids(&intervals, &ids).unwrap();
        n.merge_split();
    }

    #[test]
    fn test_merge_split_non_remains_overlapping() {
        let intervals = vec![
            15489989..15490131,
            15490871..15490947,
            15492485..15492564,
            15490131..15490871,
            15490947..15492485,
            15489989..15490131,
            15490871..15490947,
            15492485..15492564,
            15541435..15541607,
            15558445..15558626,
        ];
        let ids = vec![5, 5, 5, 9, 9, 5, 5, 5, 5, 5];
        let n = IntervalSet::new_with_ids(&intervals, &ids)
            .unwrap()
            .merge_split();
        assert!(!n.any_overlapping());
    }

    #[test]
    fn test_overlap_status() {
        let intervals = vec![
            0..100,
            50..150,
            150..200,
            201..230,
            230..250,
            249..500,
            550..600,
        ];
        let n = IntervalSet::new(&intervals).unwrap();
        assert_eq!(
            n.overlap_status(),
            vec![true, true, false, false, true, true, false]
        );
    }

    #[test]
    fn test_has_overlap_u64() {
        let r = vec![0..5, 10..15];
        let n = IntervalSetGeneric::<u64>::new(&r).unwrap();
        assert!(n.has_overlap(&(3..4)).unwrap());
        assert!(n.has_overlap(&(5..20)).unwrap());
        assert!(!n.has_overlap(&(6..10)).unwrap());
        assert!(!n.has_overlap(&(100..110)).unwrap());
        assert!(!n.has_overlap(&(3..3)).unwrap());
    }

    #[test]
    fn test_has_overlap_i128() {
        let r = vec![-50..-40, 10..15];
        let n = IntervalSetGeneric::<i128>::new(&r).unwrap();
        assert!(n.has_overlap(&(-45..46)).unwrap());
        assert!(n.has_overlap(&(-50..-49)).unwrap());
        assert!(!n.has_overlap(&(-39..-38)).unwrap());
        assert!(!n.has_overlap(&(-40..-39)).unwrap());
        assert!(n.has_overlap(&(5..20)).unwrap());
        assert!(!n.has_overlap(&(6..10)).unwrap());
        assert!(!n.has_overlap(&(100..110)).unwrap());
        assert!(!n.has_overlap(&(3..3)).unwrap());
    }

    #[test]
    fn test_filter_overlapping_split_basic() {
        let a = vec![0..100];
        let ids = [1];
        let iv_a = IntervalSet::new_with_ids(&a, &ids).unwrap();
        let b = vec![0..100];
        let other = IntervalSet::new(&b).unwrap();
        let c = iv_a.filter_to_overlapping_and_split(&other);
        assert_eq!(c.intervals, vec![0..100]);
        assert_eq!(c.ids, vec![vec![1]]);
    }

    #[test]
    fn test_filter_overlapping_split_basic2() {
        let a = vec![0..100];
        let ids = [1];
        let iv_a = IntervalSet::new_with_ids(&a, &ids).unwrap();
        let b = vec![25..50];
        let other = IntervalSet::new(&b).unwrap();
        let c = iv_a.filter_to_overlapping_and_split(&other);
        assert_eq!(c.intervals, vec![25..50]);
        assert_eq!(c.ids, vec![vec![1]]);
    }
    #[test]
    fn test_filter_overlapping_split() {
        let a = vec![0..100, 200..300, 300..450, 490..510];
        let ids = [1, 2, 3, 4];
        let iv_a = IntervalSet::new_with_ids(&a, &ids).unwrap();
        let b = vec![0..50, 50..60, 75..99, 350..500];
        let other = IntervalSet::new(&b).unwrap();
        let c = iv_a.filter_to_overlapping_and_split(&other);
        assert_eq!(c.intervals, vec![0..60, 75..99, 350..450, 490..500]);
        assert_eq!(
            c.ids,
            vec![vec![1], vec![1],  vec![3], vec![4]]
        );
    }

    #[test]
    fn test_filter_overlapping_split_2() {
        let a = vec![0..100, 100..300, 300..450, 490..510];
        let ids = [1, 2, 3, 4];
        let iv_a = IntervalSet::new_with_ids(&a, &ids).unwrap();
        let b = vec![0..50, 50..60, 75..110, 350..500];
        let other = IntervalSet::new(&b).unwrap();
        let c = iv_a.filter_to_overlapping_and_split(&other);
        assert_eq!(c.intervals, vec![0..60, 75..100, 100..110, 350..450, 490..500]);
        assert_eq!(
            c.ids,
            vec![vec![1], vec![1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn test_filter_overlapping_split_doc() {
        let a = vec![10..20];
        let ids = [99];
        let iv_a = IntervalSet::new_with_ids(&a, &ids).unwrap();
        let b = vec![5..11, 15..17, 18..30];
        let other = IntervalSet::new(&b).unwrap();
        let c = iv_a.filter_to_overlapping_and_split(&other);
        assert_eq!(c.intervals, vec![10..11, 15..17, 18..20]);
        assert_eq!(
            c.ids,
            vec![vec![99], vec![99], vec![99]]
        );
    }

}
