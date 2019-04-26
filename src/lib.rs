#![feature(nll)]
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
            .filter(|(idx, _value)| keep[*idx as usize])
            .map(|(_idx, value)| value.clone())
            .collect()
    }
}

/// IntervalSet
///
/// A collection of Range<u32> and associated ids (u32).
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
pub struct IntervalSet {
    intervals: Vec<Range<u32>>,
    ids: Vec<Vec<u32>>,
    root: Option<IntervalSetEntry>,
}

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

impl Clone for IntervalSet {
    fn clone(&self) -> IntervalSet {
        IntervalSet {
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

trait IntervalCollector {
    fn collect(&mut self, iset: &IntervalSet, no: u32);
}

struct VecIntervalCollector {
    intervals: Vec<Range<u32>>,
    ids: Vec<Vec<u32>>,
}

impl VecIntervalCollector {
    fn new() -> VecIntervalCollector {
        VecIntervalCollector {
            intervals: Vec::new(),
            ids: Vec::new(),
        }
    }
}

impl IntervalCollector for VecIntervalCollector {
    fn collect(&mut self, iset: &IntervalSet, no: u32) {
        self.intervals.push(iset.intervals[no as usize].clone());
        self.ids.push(iset.ids[no as usize].clone());
    }
}
struct TagIntervalCollector {
    hit: Vec<bool>,
}

impl TagIntervalCollector {
    fn new(iset: &IntervalSet) -> TagIntervalCollector {
        TagIntervalCollector {
            hit: vec![false; iset.len()],
        }
    }
}

impl IntervalCollector for TagIntervalCollector {
    fn collect(&mut self, _iset: &IntervalSet, no: u32) {
        self.hit[no as usize] = true;
    }
}

/// nclists are based on sorting the intervals by (start, -end)
#[allow(clippy::needless_return)]
fn nclist_range_sort(a: &Range<u32>, b: &Range<u32>) -> Ordering {
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

impl IntervalSet {
    /// Create an IntervalSet without supplying ids
    ///
    /// ids will be 0..n in the order of the *sorted* intervals
    pub fn new(intervals: &[Range<u32>]) -> IntervalSet {
        for r in intervals {
            if r.start >= r.end {
                panic!("Negative interval");
            }
        }
        let mut iv = intervals.to_vec();
        iv.sort_unstable_by(nclist_range_sort);
        let count = iv.len();
        IntervalSet {
            intervals: iv,
            ids: (0..count).map(|x| vec![x as u32]).collect(),
            root: None,
        }
    }

    /// Create an IntervalSet
    ///
    /// Ids may be non-unique
    /// This consumes both the intervals and ids
    /// which should safe an allocation in the most common use case
    pub fn new_with_ids(intervals: &[Range<u32>], ids: &[u32]) -> IntervalSet {
        for r in intervals {
            if r.start >= r.end {
                panic!("Negative interval");
            }
        }
        if intervals.len() != ids.len() {
            panic!("Intervals and ids had differing lengths");
        }
        let mut idx: Vec<usize> = (0..intervals.len()).collect();
        idx.sort_unstable_by(|idx_a, idx_b| {
            nclist_range_sort(&intervals[*idx_a], &intervals[*idx_b])
        });
        let mut out_iv: Vec<Range<u32>> = Vec::with_capacity(intervals.len());
        let mut out_ids: Vec<Vec<u32>> = Vec::with_capacity(intervals.len());
        for ii in 0..idx.len() {
            out_iv.push(intervals[idx[ii]].clone());
            out_ids.push(vec![ids[idx[ii]]]);
        }
        IntervalSet {
            intervals: out_iv,
            ids: out_ids,
            root: None,
        }
    }

    /// filter this interval set by a bool vec, true are kept
    fn new_filtered(&self, keep: &[bool]) -> IntervalSet {
        IntervalSet {
            intervals: self.intervals.filter_by_bools(&keep),
            ids: self.ids.filter_by_bools(&keep),
            root: None,
        }
    }

    /// used by the merge functions to bypass the sorting and checking
    /// on already sorted & checked intervals
    fn new_presorted(intervals: Vec<Range<u32>>, ids: Vec<Vec<u32>>) -> IntervalSet {
        IntervalSet {
            intervals,
            ids,
            root: None,
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
            std::iter::Enumerate<std::slice::Iter<'_, std::ops::Range<u32>>>,
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
    fn ensure_nclist(&mut self) {
        if self.root.is_none() {
            let mut root = IntervalSetEntry {
                no: -1,
                children: Vec::new(),
            };
            self.build_tree(&mut root, &mut self.intervals.iter().enumerate().peekable());
            self.root = Some(root);
        }
    }

    fn depth_first_search<T: IntervalCollector>(
        &self,
        node: &IntervalSetEntry,
        query: &Range<u32>,
        collector: &mut T,
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
            collector.collect(&self, next.no as u32);
            if !next.children.is_empty() {
                self.depth_first_search(next, query, collector);
            }
        }
    }
    /// Is there any interval overlapping with the query?
    pub fn has_overlap(&mut self, query: &Range<u32>) -> bool {
        if query.start > query.end {
            panic!("invalid interval end < start");
        }
        self.ensure_nclist();
        //has overlap is easy because all we have to do is scan the first level
        let root = &self.root.as_ref();
        let children = &root.unwrap().children[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first = children
            .upper_bound_by_key(&query.start, |entry| self.intervals[entry.no as usize].end);
        if first == children.len() {
            // ie no entry larger...
            return false;
        }
        let next = &self.intervals[first];
        next.overlaps(&query)
    }

    /// create an iterator over ```(Range<u32>, &vec![id])``` tuples.
    pub fn iter(
        &self,
    ) -> std::iter::Zip<std::slice::Iter<'_, std::ops::Range<u32>>, std::slice::Iter<'_, Vec<u32>>>
    {
        self.intervals.iter().zip(self.ids.iter())
    }

    /// retrieve a new IntervalSet with all intervals overlapping the query
    pub fn query_overlapping(&mut self, query: &Range<u32>) -> IntervalSet {
        self.ensure_nclist();
        let mut collector = VecIntervalCollector::new();
        self.depth_first_search(self.root.as_ref().unwrap(), &query, &mut collector);
        IntervalSet::new_presorted(collector.intervals, collector.ids)
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
        for (ii, (next, last)) in self.intervals.iter().skip(1).zip(self.intervals.iter()).enumerate() {
            if last.overlaps(next) {
                result[ii] = true; // ii starts at 0
                result[ii+1] = true;
            }
        }
        result
    }

    /// does this IntervalSet contain nested intervals?
    pub fn any_nested(&mut self) -> bool {
        self.ensure_nclist();
        for entry in self.root.as_ref().unwrap().children.iter() {
            if !entry.children.is_empty() {
                return true;
            }
        }
        false
    }

    /// remove intervals that have the same coordinates
    ///
    /// Ids are **not** merged, the first set is being kept
    pub fn remove_duplicates(&self) -> IntervalSet {
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
    pub fn remove_empty(&self) -> IntervalSet {
        let keep: Vec<bool> = self.intervals.iter().map(|r| r.start != r.end).collect();
        self.new_filtered(&keep)
    }

    /// Merge overlapping & nested intervals to their outer bounds
    ///
    /// Examples:
    /// - 0..15, 10..20 -> 0..20
    /// - 0..20, 3..5 -> 0..20
    pub fn merge_hull(&self) -> IntervalSet {
        let mut new_intervals: Vec<Range<u32>> = Vec::new();
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
                    this_ids.extend_from_slice(&next.1);
                    it.next(); // consume that one!
                } else {
                    break;
                }
            }
            new_intervals.push(this_iv);
            this_ids.sort();
            new_ids.push(this_ids)
        }
        IntervalSet::new_presorted(new_intervals, new_ids)
    }

    /// Merge intervals that are butted up against each other
    ///
    ///This first induces a merge_hull()!
    ///
    /// Examples:
    /// - 0..15, 15..20 -> 0..20
    /// - 0..15, 16..20, 20..30 > 0..15, 16..30
    pub fn merge_connected(&self) -> IntervalSet {
        let hull = self.merge_hull();
        let mut new_intervals: Vec<Range<u32>> = Vec::new();
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
                    this_ids.extend_from_slice(&next.1);
                    it.next(); // consume that one!
                } else {
                    break;
                }
            }
            new_intervals.push(this_iv);
            this_ids.sort();
            new_ids.push(this_ids)
        }
        IntervalSet::new_presorted(new_intervals, new_ids)
    }

    /// Remove all intervals that are overlapping or nested
    /// by simply dropping them.
    ///
    /// Examples:
    /// - 0..20, 10..15, 20..35, 30..40, 40..50 -> 40..50
    ///
    /// Ids of the remaining intervals are unchanged
    pub fn merge_drop(&self) -> IntervalSet {
        let mut keep = vec![true; self.len()];
        let mut last_stop = 0;
        for ii in 0..self.len() {
            if self.intervals[ii].start < last_stop {
                keep[ii] = false;
                keep[ii - 1] = false;;
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
    pub fn merge_split(&mut self) -> IntervalSet {
        #[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug)]
        enum SiteKind {
            End,
            Start,
        }
        #[derive(Debug)]
        struct Site {
            pos: u32,
            kind: SiteKind,
            id: Vec<u32>,
        }
        let mut sites: Vec<Site> = Vec::new();
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
                if (last.kind == SiteKind::Start)
                    || ((last.kind == SiteKind::End) && (next.kind == SiteKind::End))
                {
                    if last.pos != next.pos {
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
        IntervalSet {
            intervals: new_intervals,
            ids: new_ids,
            root: None,
        }
    }

    /// find the interval with the closest start to the left of pos
    /// None if there are no intervals to the left of pos
    pub fn find_closest_start_left(&mut self, pos: u32) -> Option<(Range<u32>, Vec<u32>)> {
        let first = self.intervals.upper_bound_by_key(&pos, |entry| entry.start);
        if first == 0 {
            return None;
        }
        let prev = first - 1;
        Some((self.intervals[prev].clone(), self.ids[prev].clone()))
    }

    /// find the interval with the closest start to the right of pos
    /// None if there are no intervals to the right of pos
    pub fn find_closest_start_right(&mut self, pos: u32) -> Option<(Range<u32>, Vec<u32>)> {
        let first = self
            .intervals
            .upper_bound_by_key(&pos, |entry| entry.start + 1);
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
    pub fn find_closest_start(&mut self, pos: u32) -> Option<(Range<u32>, Vec<u32>)> {
        let left = self.find_closest_start_left(pos);
        let right = self.find_closest_start_right(pos);
        match (left, right) {
            (None, None) => None,
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (Some(l), Some(r)) => {
                let distance_left = (i64::from(l.0.start) - i64::from(pos)).abs();
                let distance_right = (i64::from(r.0.start) - i64::from(pos)).abs();
                if distance_left <= distance_right {
                    Some(l)
                } else {
                    Some(r)
                }
            }
        }
    }

    /// how many units does this IntervalSet cover
    pub fn covered_units(&mut self) -> u32 {
        let merged = self.merge_hull();
        let mut total = 0;
        for iv in merged.intervals.iter() {
            total += iv.end - iv.start;
        }
        total
    }

    /// What is the mean size of the intervals
    #[allow(clippy::cast_lossless)]
    pub fn mean_interval_size(&self) -> f64 {
        let mut total = 0;
        for iv in self.intervals.iter() {
            total += iv.end - iv.start;
        }
        total as f64 / self.len() as f64
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
    pub fn invert(&self, lower_bound: u32, upper_bound: u32) -> IntervalSet {
        let mut new_intervals: Vec<Range<u32>> = Vec::new();
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
        IntervalSet::new_presorted(new_intervals, new_ids)
    }

    /// Filter to those intervals that have an overlap in other.
    pub fn filter_to_overlapping(&mut self, other: &mut IntervalSet) -> IntervalSet {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        self.ensure_nclist();
        let mut collector = TagIntervalCollector::new(&self);
        for q in other.intervals.iter() {
            self.depth_first_search(self.root.as_ref().unwrap(), q, &mut collector);
        }
        self.new_filtered(&collector.hit)
    }

    /// Filter to those intervals that have no overlap in other.
    pub fn filter_to_non_overlapping(&mut self, other: &mut IntervalSet) -> IntervalSet {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here

        let keep: Vec<bool> = self
            .intervals
            .iter()
            .enumerate()
            .map(|(_ii, iv)| !other.has_overlap(iv))
            .collect();
        self.new_filtered(&keep)
    }

    /// Filter to those intervals that have an overlap in at least k others.
    pub fn filter_to_overlapping_k_others(
        &mut self,
        others: &[&IntervalSet],
        min_k: u32,
    ) -> IntervalSet {
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
        &mut self,
        others: &[&IntervalSet],
        max_k: u32,
    ) -> IntervalSet {
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
    pub fn union(&self, others: Vec<&IntervalSet>) -> IntervalSet {
        let mut new_intervals: Vec<Range<u32>> = Vec::new();
        new_intervals.extend_from_slice(&self.intervals);
        for o in others {
            new_intervals.extend_from_slice(&o.intervals);
        }
        IntervalSet::new(&new_intervals[..])
    }

    /// find the highest interval.end in our data set
    fn _highest_end(&self) -> Option<u32> {
        match &self.root {
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

    fn _count_overlapping(&mut self, others: &[&IntervalSet]) -> Vec<u32> {
        self.ensure_nclist();
        let mut counts: Vec<u32> = vec![0; self.len()];
        for o in others {
            let mut collector = TagIntervalCollector::new(&self);
            for q in o.intervals.iter() {
                self.depth_first_search(self.root.as_ref().unwrap(), q, &mut collector);
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

impl Eq for IntervalSet {}
impl PartialEq for IntervalSet {
    fn eq(&self, other: &IntervalSet) -> bool {
        (self.intervals == other.intervals) && (self.ids == other.ids)
    }
}

/// Extend Range functionality with some often used bool functions
pub trait RangePlus<T> {
    /// Does this interval overlap the other one?
    fn overlaps(&self, other: &Range<T>) -> bool;
}

impl RangePlus<u32> for Range<u32> {
    fn overlaps(&self, other: &Range<u32>) -> bool {
        (self.start < other.end && other.start < self.end && other.start < other.end)
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use crate::IntervalSet;
    use std::ops::Range;
    #[test]
    fn test_has_overlap() {
        let r = vec![0..5, 10..15];
        let mut n = IntervalSet::new(&r);
        assert!(n.has_overlap(&(3..4)));
        assert!(n.has_overlap(&(5..20)));
        assert!(!n.has_overlap(&(6..10)));
        assert!(!n.has_overlap(&(100..110)));
        assert!(!n.has_overlap(&(3..3)));

        let r2 = vec![0..15, 0..6];
        let mut n = IntervalSet::new(&r2);
        assert!(n.has_overlap(&(3..4)));
        assert!(n.has_overlap(&(5..20)));
        assert!(n.has_overlap(&(6..10)));
        assert!(!n.has_overlap(&(20..30)));

        let r2 = vec![100..150, 30..40, 200..400];
        let mut n = IntervalSet::new(&r2);
        assert!(n.has_overlap(&(101..102)));
        assert!(n.has_overlap(&(149..150)));
        assert!(n.has_overlap(&(39..99)));
        assert!(n.has_overlap(&(29..99)));
        assert!(n.has_overlap(&(19..99)));
        assert!(!n.has_overlap(&(0..5)));
        assert!(!n.has_overlap(&(0..29)));
        assert!(!n.has_overlap(&(0..30)));
        assert!(n.has_overlap(&(0..31)));
        assert!(!n.has_overlap(&(40..41)));
        assert!(!n.has_overlap(&(40..99)));
        assert!(!n.has_overlap(&(40..100)));
        assert!(n.has_overlap(&(40..101)));
        assert!(n.has_overlap(&(399..400)));
        assert!(!n.has_overlap(&(400..4000)));
    }
    #[test]
    fn test_iter() {
        let n = IntervalSet::new(&vec![100..150, 30..40, 200..400, 250..300]);
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
        let mut n = IntervalSet::new(&vec![100..150, 30..40, 200..400, 250..300]);
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
        let mut n = IntervalSet::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&vec![0..5, 0..105]));
        assert_eq!(c.intervals, vec![30..40, 100..150]);
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&vec![500..600, 550..700]));
        assert!(c.is_empty());
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&vec![45..230]));
        assert_eq!(c.intervals, vec![100..150, 200..400]);
        let c = n.filter_to_overlapping(&mut IntervalSet::new(&vec![45..101, 101..230]));
        assert_eq!(c.intervals, vec![100..150, 200..400]);
    }

    #[test]
    fn test_query_multiple_non_overlapping() {
        let mut n = IntervalSet::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&vec![0..5, 0..105]));
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&vec![500..600, 550..700]));
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&vec![0..600]));
        assert!(c.is_empty());
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&vec![45..230]));
        assert_eq!(c.intervals, vec![30..40, 250..300]);
        let c = n.filter_to_non_overlapping(&mut IntervalSet::new(&vec![45..101, 101..230]));
        assert_eq!(c.intervals, vec![30..40, 250..300]);
    }

    #[test]
    fn test_any_overlapping() {
        let n = IntervalSet::new(&vec![100..150]);
        assert!(!n.any_overlapping());
        let n = IntervalSet::new(&vec![100..150, 200..300]);
        assert!(!n.any_overlapping());
        let n = IntervalSet::new(&vec![100..150, 150..300]);
        assert!(!n.any_overlapping());
        let n = IntervalSet::new(&vec![100..151, 150..300]);
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&vec![100..151, 105..110]);
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&vec![100..151, 105..110, 0..1000]);
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&vec![100..150, 150..210, 0..1000]);
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&vec![100..150, 150..210, 0..130]);
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&vec![100..150, 150..210, 150..250]);
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&vec![100..150, 150..210, 149..250]);
        assert!(n.any_overlapping());
        let n = IntervalSet::new(&vec![100..150, 150..210, 209..250]);
        assert!(n.any_overlapping());
    }

    #[test]
    fn test_any_nested() {
        assert!(!IntervalSet::new(&vec![]).any_nested());;
        assert!(!IntervalSet::new(&vec![100..150]).any_nested());;
        assert!(!IntervalSet::new(&vec![100..150, 150..300]).any_nested());;
        assert!(!IntervalSet::new(&vec![100..151, 150..300]).any_nested());;
        assert!(IntervalSet::new(&vec![100..151, 150..300, 100..130]).any_nested());;
        assert!(IntervalSet::new(&vec![100..151, 150..300, 0..1000]).any_nested());;
    }

    #[test]
    fn test_remove_duplicates() {
        let n = IntervalSet::new(&vec![100..150]).remove_duplicates();
        assert!(!n.any_overlapping());
        assert_eq!(n.len(), 1);

        let n = IntervalSet::new(&vec![30..40, 30..40, 100..150]).remove_duplicates();
        assert!(!n.any_overlapping());
        assert_eq!(n.len(), 2);
        let n = IntervalSet::new(&vec![30..40, 30..40, 35..150]).remove_duplicates();
        assert_eq!(n.len(), 2);
        let n = IntervalSet::new_with_ids(
            &vec![30..40, 30..40, 35..150, 35..150, 36..38],
            &[55, 56, 57, 58, 59],
        )
        .remove_duplicates();
        assert_eq!(n.len(), 3);
        dbg!(&n);
        assert_eq!(n.ids, vec![vec![55], vec![57], vec![59]]);
    }

    #[test]
    fn test_merge_hull() {
        let n = IntervalSet::new(&vec![100..150, 120..180, 110..115, 200..201]).merge_hull();
        assert_eq!(n.intervals, vec![100..180, 200..201]);
        assert_eq!(n.ids, vec![vec![0, 1, 2], vec![3]]);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&vec![100..150]).merge_hull();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&vec![100..150, 120..180]).merge_hull();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![100..180]);
        assert_eq!(n.ids, vec![vec![0, 1]]);

        let n = IntervalSet::new(&vec![100..150, 120..180, 110..115]).merge_hull();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![100..180]);
        assert_eq!(n.ids, vec![vec![0, 1, 2]]);

        let n = IntervalSet::new_with_ids(
            &vec![300..400, 400..450, 450..500, 510..520],
            &[20, 10, 30, 40],
        );
        assert_eq!(n.intervals, vec![300..400, 400..450, 450..500, 510..520]);
    }

    #[test]
    fn test_merge_drop() {
        let n = IntervalSet::new(&vec![]).merge_drop();
        assert_eq!(n.len(), 0);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&vec![100..150]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());

        let n = IntervalSet::new(&vec![100..150, 120..180]).merge_drop();
        assert_eq!(n.len(), 0);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![]);
        assert_eq!(n.ids, Vec::<Vec<u32>>::new());

        let n = IntervalSet::new(&vec![100..150, 120..180, 200..250]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![2]]);

        let n = IntervalSet::new(&vec![100..150, 120..180, 200..250, 106..110]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![3]]);

        let n =
            IntervalSet::new(&vec![100..150, 120..180, 200..250, 106..110, 80..105]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![4]]);

        let n = IntervalSet::new(&vec![
            100..150,
            120..180,
            200..250,
            106..110,
            80..105,
            30..40,
        ])
        .merge_drop();
        assert_eq!(n.len(), 2);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![30..40, 200..250]);
        assert_eq!(n.ids, vec![vec![0], vec![5]]);

        let n = IntervalSet::new(&vec![
            100..150,
            120..180,
            200..250,
            106..110,
            80..105,
            30..40,
            400..405,
        ])
        .merge_drop();
        assert_eq!(n.len(), 3);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![30..40, 200..250, 400..405]);
        assert_eq!(n.ids, vec![vec![0], vec![5], vec![6]]);
        let n = IntervalSet::new(&vec![0..20, 10..15, 20..35, 30..40, 40..50]).merge_drop();
        assert_eq!(n.intervals, vec![40..50]);
    }

    #[test]
    fn test_find_closest_start_left() {
        let mut n = IntervalSet::new(&vec![
            30..40,
            80..105,
            100..150,
            106..110,
            106..120,
            107..125,
            120..180,
            200..250,
            400..405,
        ]);
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
        let mut n = IntervalSet::new(&vec![]);
        assert!(n.find_closest_start_left(29).is_none());
    }

    #[test]
    fn test_find_closest_start_right() {
        let mut n = IntervalSet::new(&vec![
            30..40,
            80..105,
            100..150,
            106..110,
            106..120,
            107..125,
            120..180,
            200..250,
            400..405,
        ]);
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
        let mut n = IntervalSet::new(&vec![]);
        assert!(n.find_closest_start_right(29).is_none());
    }

    #[test]
    fn test_find_closest_start() {
        let mut n = IntervalSet::new(&vec![]);
        assert!(n.find_closest_start(100).is_none());
        let mut n = IntervalSet::new(&vec![100..110, 200..300]);
        assert_eq!(n.find_closest_start(0).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(100).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(149).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(150).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(151).unwrap(), (200..300, vec![1]));
        assert_eq!(n.find_closest_start(251).unwrap(), (200..300, vec![1]));
        assert_eq!(n.find_closest_start(351).unwrap(), (200..300, vec![1]));

        let mut n = IntervalSet::new(&vec![10..11, 1000..1110]);
        assert_eq!(n.find_closest_start(5).unwrap(), (10..11, vec![0]));
        let mut n = IntervalSet::new(&vec![
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
        ]);

        assert_eq!(
            n.find_closest_start(570000).unwrap(),
            (569592..570304, vec![1])
        );
    }

    #[test]
    fn test_covered_units() {
        let mut n = IntervalSet::new(&vec![]);
        assert_eq!(n.covered_units(), 0);
        let mut n = IntervalSet::new(&vec![10..100]);
        assert_eq!(n.covered_units(), 90);
        let mut n = IntervalSet::new(&vec![10..100, 200..300]);
        assert_eq!(n.covered_units(), 90 + 100);
        let mut n = IntervalSet::new(&vec![10..100, 200..300, 15..99]);
        assert_eq!(n.covered_units(), 90 + 100);
        let mut n = IntervalSet::new(&vec![10..100, 200..300, 15..99, 15..105]);
        assert_eq!(n.covered_units(), 90 + 100 + 5);
    }

    #[test]
    fn test_mean_interval_size() {
        let n = IntervalSet::new(&vec![]);
        assert!(n.mean_interval_size().is_nan());
        let n = IntervalSet::new(&vec![10..100]);
        assert_eq!(n.mean_interval_size(), 90.);
        let n = IntervalSet::new(&vec![10..100, 200..300]);
        assert_eq!(n.mean_interval_size(), (90 + 100) as f64 / 2.0);
        let n = IntervalSet::new(&vec![10..100, 200..300, 15..99]);
        assert_eq!(n.mean_interval_size(), (90 + 100 + (99 - 15)) as f64 / 3.0);
        let n = IntervalSet::new(&vec![10..100, 200..300, 15..99, 15..105]);
        assert_eq!(
            n.mean_interval_size(),
            (((100 - 10) + (300 - 200) + (99 - 15) + (105 - 15)) as f64 / 4.0)
        );
    }

    #[test]
    fn test_invert() {
        let n = IntervalSet::new(&vec![]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..100,]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = IntervalSet::new(&vec![30..40]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 40..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = IntervalSet::new(&vec![30..40, 35..38]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 40..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = IntervalSet::new(&vec![30..40, 35..38, 35..50]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 50..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = IntervalSet::new(&vec![30..40, 35..38, 35..50]).invert(40, 100);
        assert_eq!(n.intervals, vec![50..100,]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = IntervalSet::new(&vec![30..40, 35..38, 35..50, 55..60]).invert(40, 40);
        assert_eq!(n.intervals, vec![50..55]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = IntervalSet::new(&vec![30..40, 35..38, 35..50]).invert(40, 40);
        assert!(n.intervals.is_empty());
        assert!(n.ids.is_empty());
    }

    #[test]
    fn test_union() {
        let n = IntervalSet::new(&vec![]).union(vec![&IntervalSet::new(&vec![0..100])]);
        assert_eq!(n.intervals, vec![0..100]);

        let n = IntervalSet::new(&vec![0..10]).union(vec![&IntervalSet::new(&vec![0..100])]);
        assert_eq!(n.intervals, vec![0..100, 0..10]);

        let n =
            IntervalSet::new(&vec![0..10]).union(vec![&IntervalSet::new(&vec![0..100, 200..300])]);
        assert_eq!(n.intervals, vec![0..100, 0..10, 200..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![2]]);

        let n = IntervalSet::new(&vec![0..10]).union(vec![&IntervalSet::new(&vec![])]);
        assert_eq!(n.intervals, vec![0..10]);
        let n = IntervalSet::new(&vec![0..10]).union(vec![
            &IntervalSet::new(&vec![0..100]),
            &IntervalSet::new(&vec![200..300]),
        ]);
        assert_eq!(n.intervals, vec![0..100, 0..10, 200..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn test_substract() {
        let n = IntervalSet::new(&vec![])
            .filter_to_non_overlapping(&mut IntervalSet::new(&vec![0..100]));
        assert!(n.intervals.is_empty());

        let n = IntervalSet::new(&vec![0..10])
            .filter_to_non_overlapping(&mut IntervalSet::new(&vec![0..100]));
        assert!(n.intervals.is_empty());

        let n = IntervalSet::new(&vec![0..10, 100..150])
            .filter_to_non_overlapping(&mut IntervalSet::new(&vec![0..100]));
        assert_eq!(n.intervals, vec![100..150]);

        let n = IntervalSet::new(&vec![0..10, 100..150, 150..300])
            .filter_to_non_overlapping(&mut IntervalSet::new(&vec![55..101]));
        assert_eq!(n.intervals, vec![0..10, 150..300]);
        assert_eq!(n.ids, vec![vec![0], vec![2]]);

        let n = IntervalSet::new(&vec![0..10, 5..6, 100..150, 150..300])
            .filter_to_non_overlapping(&mut IntervalSet::new(&vec![55..101]));
        assert_eq!(n.intervals, vec![0..10, 5..6, 150..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![3]]);
    }

    #[test]
    fn test_filter_overlapping_multiples() {
        let mut n = IntervalSet::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_overlapping_k_others(&[&IntervalSet::new(&vec![0..5, 0..105])], 1);
        assert_eq!(c.intervals, vec![30..40, 100..150]);
        let c = n.filter_to_overlapping_k_others(&[&IntervalSet::new(&vec![0..5, 0..105])], 0);
        assert_eq!(c, n);
        let c = n.filter_to_overlapping_k_others(&[&IntervalSet::new(&vec![0..5, 0..105])], 2);
        assert!(c.is_empty());

        let c = n.filter_to_overlapping_k_others(
            &[
                &IntervalSet::new(&vec![0..35]),
                &IntervalSet::new(&vec![0..160]),
            ],
            2,
        );
        assert_eq!(c.intervals, vec![30..40,]);
        let c = n.filter_to_overlapping_k_others(
            &[
                &IntervalSet::new(&vec![0..35]),
                &IntervalSet::new(&vec![0..160]),
            ],
            1,
        );
        assert_eq!(c.intervals, vec![30..40, 100..150]);
    }

    #[test]
    fn test_filter_non_overlapping_multiples() {
        let mut n = IntervalSet::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(&[&IntervalSet::new(&vec![0..5, 0..105])], 1);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(&[&IntervalSet::new(&vec![0..5, 0..105])], 0);
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(&[&IntervalSet::new(&vec![0..5, 0..105])], 2);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);

        let c = n.filter_to_non_overlapping_k_others(
            &[
                &IntervalSet::new(&vec![0..35]),
                &IntervalSet::new(&vec![0..160]),
            ],
            2,
        );
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(
            &[
                &IntervalSet::new(&vec![0..35]),
                &IntervalSet::new(&vec![0..160]),
            ],
            1,
        );
        assert_eq!(c.intervals, vec![100..150, 200..400, 250..300]);
    }

    #[test]
    fn test_split() {
        let mut n = IntervalSet::new(&vec![0..100, 20..30]);
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..20, 20..30, 30..100]);
        assert_eq!(c.ids, vec![vec![0], vec![0, 1,], vec![0]]);

        let mut n = IntervalSet::new(&vec![0..100, 0..90, 70..95, 110..150]);
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..70, 70..90, 90..95, 95..100, 110..150]);
        assert_eq!(
            c.ids,
            vec![vec![0, 1], vec![0, 1, 2], vec![0, 2], vec![0], vec![3]]
        );
        let mut n = IntervalSet::new_with_ids(
            &vec![0..100, 0..90, 70..95, 110..150],
            &vec![100, 200, 300, 400],
        );
        let mut c = n.merge_split();
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
    }

    #[test]
    fn test_example() {
        let intervals = vec![0..20, 15..30, 50..100];
        let mut interval_set = IntervalSet::new(&intervals);
        assert_eq!(interval_set.ids, vec![vec![0], vec![1], vec![2]]); // automatic ids, use new_with_ids otherwise
        let hits = interval_set.query_overlapping(&(10..16));
        assert_eq!(hits.intervals, [0..20, 15..30]);
        let merged = hits.merge_hull();
        assert_eq!(merged.intervals, [0..30]);
        assert_eq!(merged.ids, vec![vec![0, 1]]);
    }

    #[test]
    fn test_new_with_ids_sorting() {
        let n = IntervalSet::new_with_ids(&vec![300..400, 30..40], &[20, 10]);
        assert_eq!(n.intervals, [30..40, 300..400]);
        assert_eq!(n.ids, vec![vec![10], vec![20]]);
    }

    #[test]
    fn test_merge_connectd() {
        let n = IntervalSet::new_with_ids(&vec![300..400, 400..450, 450..500], &[20, 10, 30]);
        assert_eq!(n.merge_hull().intervals, vec![300..400, 400..450, 450..500]);
        let n = n.merge_connected();
        assert_eq!(n.intervals, [300..500]);
        assert_eq!(n.ids, vec![vec![10, 20, 30],]);

        let n = IntervalSet::new_with_ids(&vec![300..400, 400..450, 451..500], &[20, 10, 30])
            .merge_connected();
        assert_eq!(n.intervals, [300..450, 451..500]);
        assert_eq!(n.ids, vec![vec![10, 20], vec![30]]);
        let n = IntervalSet::new_with_ids(
            &vec![300..400, 400..450, 451..500, 350..355],
            &[20, 10, 30, 40],
        )
        .merge_connected();
        assert_eq!(n.intervals, [300..450, 451..500]);
        assert_eq!(n.ids, vec![vec![10, 20, 40], vec![30]]);
        let n = IntervalSet::new_with_ids(
            &vec![300..400, 400..450, 450..500, 510..520],
            &[20, 10, 30, 40],
        )
        .merge_connected();
        assert_eq!(n.intervals, vec![300..500, 510..520]);
    }

    #[test]
    fn test_clone() {
        let mut n = IntervalSet::new_with_ids(&vec![300..400, 400..450, 450..500], &[20, 10, 30]);
        let n2 = n.clone();
        assert_eq!(n.intervals, n2.intervals);
        assert_eq!(n.ids, n2.ids);
        assert!(n2.root.is_none());
        n.has_overlap(&(0..1));
        assert!(n2.root.is_none());
        let n2 = n.clone();
        assert!(n2.root.is_some());
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

        let mut n = IntervalSet::new_with_ids(&intervals, &ids);
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
        let n = IntervalSet::new_with_ids(&intervals, &ids).merge_split();
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
            550..600];
        let n = IntervalSet::new(&intervals);
        assert_eq!(n.overlap_status(), vec![true, true, false, false, true, true, false] );
    }
}
