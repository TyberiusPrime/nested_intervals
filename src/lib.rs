#![feature(nll)]
use std::cmp::Ordering;
use std::ops::Range;
use superslice::*;


trait FilterByBools<T>{
    fn filter_by_bools(&self, keep: &Vec<bool>) -> Vec<T>; 
}

impl <T> FilterByBools<T> for Vec<T>
where T: Clone
{
    fn filter_by_bools(&self, keep: &Vec<bool>) -> Vec<T> {
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

#[derive(Debug)]
struct NCList {
    intervals: Vec<Range<u32>>,
    ids: Vec<u32>,
    root: Option<NCListEntry>,
}

#[derive(Debug)]
pub struct NCListEntry {
    no: i32,
    children: Vec<NCListEntry>,
}

type NCListResult = (Range<u32>, u32);

fn nclist_range_sort(a: &Range<u32>, b: &Range<u32>) -> Ordering {
    if a.start < b.start {
        return Ordering::Less;
    } else if a.start > b.start {
        return Ordering::Greater;
    } else {
        if a.end > b.end {
            return Ordering::Less; // the magic trick to get contained intervals
        } else if a.end < b.end {
            return Ordering::Greater;
        } else {
            return Ordering::Equal;
        }
    }
}

impl NCList {
    pub fn new(intervals: &[Range<u32>]) -> NCList {
        for r in intervals {
            if r.start >= r.end {
                panic!("Negative interval");
            }
        }
        let mut iv = intervals.to_vec();
        iv.sort_unstable_by(nclist_range_sort);
        let count = iv.len();
        NCList {
            intervals: iv,
            ids: (0..count).map(|x| x as u32).collect(),
            root: None,
        }
    }
    pub fn new_with_ids(intervals: &[Range<u32>], ids: &[u32]) -> NCList {
        for r in intervals {
            if r.start >= r.end {
                panic!("Negative interval");
            }
        }
        if intervals.len() != ids.len() {
            panic!("Intervals and ids had differing lengths");
        }
        let mut iv = intervals.to_vec();
        iv.sort_unstable_by(nclist_range_sort);
        let ids = ids.to_vec();
        NCList {
            intervals: iv,
            ids: ids,
            root: None,
        }
    }

    fn len(&self) -> usize {
        self.intervals.len()
    }

    fn build_tree(
        &self,
        parent: &mut NCListEntry,
        it: &mut std::iter::Peekable<
            std::iter::Enumerate<std::slice::Iter<'_, std::ops::Range<u32>>>,
        >,
    ) {
        loop {
            match it.peek() {
                Some((_, next)) => {
                    if (parent.no != -1) && (next.start > self.intervals[parent.no as usize].end) {
                        return;
                    }
                    let (ii, r) = it.next().unwrap();
                    if r.start > r.end {
                        panic!("invalid interval end < start");
                    }
                    let entry = NCListEntry {
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

    fn ensure_nclist(&mut self) {
        if self.root.is_none() {
            let mut root = NCListEntry {
                no: -1,
                children: Vec::new(),
            };
            self.build_tree(&mut root, &mut self.intervals.iter().enumerate().peekable());
            self.root = Some(root);
        }
    }

    pub fn has_overlap(&mut self, query: Range<u32>) -> bool {
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
        return next.overlaps(&query);
    }

    pub fn iter(
        &self,
    ) -> std::iter::Zip<std::slice::Iter<'_, std::ops::Range<u32>>, std::slice::Iter<'_, u32>> {
        self.intervals.iter().zip(self.ids.iter())
    }

    pub fn query_overlapping(&mut self, query: Range<u32>) -> Vec<NCListResult> {
        self.ensure_nclist();
        let mut res: Vec<NCListResult> = Vec::new();
        self._query_overlapping(&self.root.as_ref().unwrap(), &query, &mut res);
        return res;
    }

    pub fn query_overlapping_multiple(&mut self, query: &[Range<u32>]) -> Vec<NCListResult> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        self.ensure_nclist();
        let mut keep = vec![false; self.len()];
        for q in query {
            self._tag_overlapping(&self.root.as_ref().unwrap(), &q, &mut keep);
        }
        return self
            .intervals
            .iter()
            .zip(self.ids.iter())
            .zip(keep.iter())
            .filter(|(_value, do_keep)| **do_keep)
            .map(|(value, _do_keep)| (value.0.clone(), *value.1))
            .collect();
    }

    pub fn query_non_overlapping_multiple(&mut self, query: &[Range<u32>]) -> Vec<NCListResult> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        self.ensure_nclist();
        let mut keep = vec![false; self.len()];
        for q in query {
            self._tag_overlapping(&self.root.as_ref().unwrap(), &q, &mut keep);
        }
        return self
            .intervals
            .iter()
            .zip(self.ids.iter())
            .zip(keep.iter())
            .filter(|(_value, do_keep)| !**do_keep)
            .map(|(value, _do_keep)| (value.0.clone(), *value.1))
            .collect();
    }

    pub fn any_overlapping(&self) -> bool {
        for (next, last) in self.intervals.iter().skip(1).zip(self.intervals.iter()) {
            if last.overlaps(next) {
                return true;
            }
        }
        false
    }

    pub fn remove_duplicates(&self) -> NCList {
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
        NCList {
            intervals: self.intervals.filter_by_bools(&keep),
            ids: self.ids.filter_by_bools(&keep),
            root: None,
        }
    }

    fn _query_overlapping(
        &self,
        node: &NCListEntry,
        query: &Range<u32>,
        collector: &mut Vec<NCListResult>,
    ) {
        let children = &node.children[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first = children
            .upper_bound_by_key(&query.start, |entry| self.intervals[entry.no as usize].end);
        if first == children.len() {
            return;
        }
        for next in &children[first..] {
            let next_iv = &self.intervals[next.no as usize];
            let next_id = self.ids[next.no as usize];
            if !next_iv.overlaps(query) {
                return;
            }
            collector.push((next_iv.clone(), next_id));
            if !next.children.is_empty() {
                self._query_overlapping(next, query, collector);
            }
        }
    }
    fn _tag_overlapping(&self, node: &NCListEntry, query: &Range<u32>, tags: &mut Vec<bool>) {
        let children = &node.children[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
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
            tags[next.no as usize] = true;
            if !next.children.is_empty() {
                self._tag_overlapping(next, query, tags);
            }
        }
    }
}

trait ToIntervalVec {
    fn to_interval_vec(&self) -> Vec<Range<u32>>;
}
impl ToIntervalVec for Vec<NCListResult> {
    fn to_interval_vec(&self) -> Vec<Range<u32>> {
        self.iter().map(|(iv, _id)| iv.clone()).collect()
    }
}

trait RangePlus<T> {
    fn overlaps(&self, other: &Range<T>) -> bool;
    fn is_to_the_rigth_of(&self, other: &Range<T>) -> bool;
}

impl RangePlus<u32> for Range<u32> {
    fn overlaps(&self, other: &Range<u32>) -> bool {
        (self.start < other.end && other.start < self.end)
    }
    fn is_to_the_rigth_of(&self, other: &Range<u32>) -> bool {
        self.end > other.start
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use crate::{NCList, NCListResult, ToIntervalVec};
    use std::ops::Range;
    #[test]
    fn test_has_overlap() {
        let r = vec![0..5, 10..15];
        let mut n = NCList::new(&r);
        assert!(n.has_overlap(3..4));
        assert!(n.has_overlap(5..20));
        assert!(!n.has_overlap(6..10));
        assert!(!n.has_overlap(100..110));

        let r2 = vec![0..15, 0..6];
        let mut n = NCList::new(&r2);
        assert!(n.has_overlap(3..4));
        assert!(n.has_overlap(5..20));
        assert!(n.has_overlap(6..10));
        assert!(!n.has_overlap(20..30));

        let r2 = vec![100..150, 30..40, 200..400];
        let mut n = NCList::new(&r2);
        assert!(n.has_overlap(101..102));
        assert!(n.has_overlap(149..150));
        assert!(n.has_overlap(39..99));
        assert!(n.has_overlap(29..99));
        assert!(n.has_overlap(19..99));
        assert!(!n.has_overlap(0..5));
        assert!(!n.has_overlap(0..29));
        assert!(!n.has_overlap(0..30));
        assert!(n.has_overlap(0..31));
        assert!(!n.has_overlap(40..41));
        assert!(!n.has_overlap(40..99));
        assert!(!n.has_overlap(40..100));
        assert!(n.has_overlap(40..101));
        assert!(n.has_overlap(399..400));
        assert!(!n.has_overlap(400..4000));
    }
    #[test]
    fn test_iter() {
        let n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c: Vec<(&Range<u32>, &u32)> = n.iter().collect();
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
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_overlapping(0..5);
        assert!(c.is_empty());
        let c: Vec<NCListResult> = n.query_overlapping(0..31);
        assert!(c.to_interval_vec() == vec![30..40]);
        let c: Vec<NCListResult> = n.query_overlapping(200..250);
        assert!(c.to_interval_vec() == vec![200..400]);
        let c: Vec<NCListResult> = n.query_overlapping(200..251);
        assert!(c.to_interval_vec() == vec![200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_overlapping(0..1000);
        dbg!(&c);
        assert!(c.to_interval_vec() == vec![30..40, 100..150, 200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_overlapping(401..1000);
        assert!(c.is_empty());
    }
    #[test]
    fn test_query_multiple() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[0..5, 0..105]);
        assert!(c.to_interval_vec() == vec![30..40, 100..150]);
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[500..600, 550..700]);
        assert!(c.is_empty());
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[45..230]);
        assert!(c.to_interval_vec() == vec![100..150, 200..400]);
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[45..101, 101..230]);
        assert!(c.to_interval_vec() == vec![100..150, 200..400]);
    }
    #[test]
    fn test_query_multiple_non_overlapping() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_non_overlapping_multiple(&[0..5, 0..105]);
        assert!(c.to_interval_vec() == vec![200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_non_overlapping_multiple(&[500..600, 550..700]);
        assert!(c.to_interval_vec() == vec![30..40, 100..150, 200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_non_overlapping_multiple(&[0..600]);
        assert!(c.is_empty());
        let c: Vec<NCListResult> = n.query_non_overlapping_multiple(&[45..230]);
        assert!(c.to_interval_vec() == vec![30..40, 250..300]);
        let c: Vec<NCListResult> = n.query_non_overlapping_multiple(&[45..101, 101..230]);
        assert!(c.to_interval_vec() == vec![30..40, 250..300]);
    }

    #[test]
    fn test_any_overlapping() {
        let n = NCList::new(&vec![100..150]);
        assert!(!n.any_overlapping());
        let n = NCList::new(&vec![100..150, 200..300]);
        assert!(!n.any_overlapping());
        let n = NCList::new(&vec![100..150, 150..300]);
        assert!(!n.any_overlapping());
        let n = NCList::new(&vec![100..151, 150..300]);
        assert!(n.any_overlapping());
        let n = NCList::new(&vec![100..151, 105..110]);
        assert!(n.any_overlapping());
        let n = NCList::new(&vec![100..151, 105..110, 0..1000]);
        assert!(n.any_overlapping());
        let n = NCList::new(&vec![100..150, 150..210, 0..1000]);
        assert!(n.any_overlapping());
        let n = NCList::new(&vec![100..150, 150..210, 0..130]);
        assert!(n.any_overlapping());
        let n = NCList::new(&vec![100..150, 150..210, 150..250]);
        assert!(n.any_overlapping());
        let n = NCList::new(&vec![100..150, 150..210, 149..250]);
        assert!(n.any_overlapping());
        let n = NCList::new(&vec![100..150, 150..210, 209..250]);
        assert!(n.any_overlapping());
    }

    #[test]
    fn test_remove_duplicates() {
        let n = NCList::new(&vec![100..150]).remove_duplicates();
        assert!(!n.any_overlapping());
        assert!(n.len() == 1);
        let n = NCList::new(&vec![30..40, 30..40, 100..150]).remove_duplicates();
        assert!(!n.any_overlapping());
        assert!(n.len() == 2);
        let n = NCList::new(&vec![30..40, 30..40, 35..150]).remove_duplicates();
        assert!(n.len() == 2);
        let n = NCList::new_with_ids(
            &vec![30..40, 30..40, 35..150, 35..150, 36..38],
            &[55,56,57,58,59]
            ).remove_duplicates();
        assert!(n.len() == 3);
        dbg!(&n);
        assert!(n.ids == vec![55,57,59]);
    }
}
