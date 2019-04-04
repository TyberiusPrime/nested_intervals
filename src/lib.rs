#![feature(nll)]
use std::cmp::Ordering;
use std::ops::Range;
use superslice::*;

struct NCList {
    intervals: Vec<Range<u32>>,
    ids: Vec<u32>,
    root: Option<NCListEntry>,
}

#[derive(Debug)]
struct NCListEntry {
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
    fn new(intervals: &[Range<u32>]) -> NCList {
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

    fn has_overlap(&mut self, query: Range<u32>) -> bool {
        if query.start > query.end {
            panic!("invalid interval end < start");
        }
        self.ensure_nclist();
        //has overlap is easy because all we have to do is scan the first level
        let root = &self.root.as_ref();
        let children = &root.unwrap().children[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first =
            children.upper_bound_by_key(&query.start, |entry| self.intervals[entry.no as usize].end);
        if first == children.len() {
            // ie no entry larger...
            return false;
        }
        let next = &self.intervals[first];
        return next.overlaps(&query);
    }

    fn iter(
        &self,
    ) -> std::iter::Zip<std::slice::Iter<'_, std::ops::Range<u32>>, std::slice::Iter<'_, u32>> {
        self.intervals.iter().zip(self.ids.iter())
    }
    fn query_overlapping(&mut self, query: Range<u32>) -> Vec<NCListResult> {
        self.ensure_nclist();
        let mut res: Vec<NCListResult> = Vec::new();
        self._query_overlapping(&self.root.as_ref().unwrap(), &query, &mut res);
        return res;
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
        let first =
            children.upper_bound_by_key(&query.start, |entry| self.intervals[entry.no as usize].end);
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
    fn _tag_overlapping(&self, node: &NCListEntry, query: &Range<u32>,tags: &mut Vec<bool>) {
        let children = &node.children[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first = 
            children.upper_bound_by_key(&query.start, |entry| self.intervals[entry.no as usize].end);
        if first == children.len() {
            return;
        }
        for next in &children[first..] {
            let next_iv = &self.intervals[next.no as usize];
            let next_id = self.ids[next.no as usize];
            if !next_iv.overlaps(query) {
                return;
            }
            tags[next.no as usize] = true;
            if !next.children.is_empty() {
                self._tag_overlapping(next, query, tags);
            }
        }
    }
    fn query_overlapping_multiple(&mut self, query: &[Range<u32>]) -> Vec<NCListResult> {
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
            .filter(|(value, do_keep)| **do_keep)
            .map(|(value, do_keep)| (value.0.clone(), *value.1)).collect();
    }
}

trait ToIntervalVec {
    fn to_interval_vec(&self) -> Vec<Range<u32>>;
}
impl ToIntervalVec for Vec<NCListResult> {
    fn to_interval_vec(&self) -> Vec<Range<u32>> {
        self.iter().map(|(iv, id)| iv.clone()).collect()
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

        let r2 = vec![10..15, 0..5];
        let mut n = NCList::new(&r2);

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
        let c: Vec<Range<u32>> = c.iter().map(|(interval, id)| (*interval).clone()).collect();
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
}
