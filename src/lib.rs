#![feature(nll)]
use std::cmp::Ordering;
use std::ops::Range;
use superslice::*;

#[derive(Debug)]
struct NCListEntry {
    start: u32,
    end: u32,
    id: u32,
    sublist: Vec<NCListEntry>,
    parents: Option<Vec<u32>>, // where did these intervals come from
}

#[derive(Debug)]
struct NCListResult {
    start: u32,
    end: u32,
    id: u32,
    parents: Option<Vec<u32>>,
}

impl NCListEntry {
    fn new(start: u32, end: u32, index: u32) -> NCListEntry {
        NCListEntry {
            start,
            end,
            sublist: Vec::new(),
            id: index,
            parents: None,
        }
    }
    //poor mans iterator...
    fn collect_into_vector(&self, collector: &mut Vec<NCListResult>) {
        for child in self.sublist.iter() {
            collector.push(child.into());
            child.collect_into_vector(collector);
        }
    }

    fn query_overlapping(&self, query: &Range<u32>,collector: &mut Vec<NCListResult>) {
        let sublist = &self.sublist[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first = sublist.upper_bound_by_key(&query.start, |entry| entry.end);
        if first == sublist.len() {
            return;
        }
        for next in &sublist[first..] {
            if query.start > next.end {
                return;
            }
            if next.start >= query.end {
                return;
            }
            collector.push(next.into());
            if !next.sublist.is_empty() {
                next.query_overlapping(query, collector);
            }
        }
    }

    fn tag_overlapping(&self, query: &Range<u32>,tags: &mut Vec<bool>) {
        let sublist = &self.sublist[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first = sublist.upper_bound_by_key(&query.start, |entry| entry.end);
        if first == sublist.len() {
            return;
        }
        for next in &sublist[first..] {
            if query.start > next.end {
                return;
            }
            if next.start >= query.end {
                return;
            }
            tags[next.id as usize] = true;
            if !next.sublist.is_empty() {
                next.tag_overlapping(query, tags);
            }
        }
    }
    fn retrieve_tagged(&self, keep: &Vec<bool>, collector: &mut Vec<NCListResult>){
        for child in self.sublist.iter() {
            if keep[child.id as usize] {
                collector.push(child.into());
            }
            child.retrieve_tagged(keep, collector);
        }
    }
}

impl NCListResult {
    fn new(entry: &NCListEntry) -> NCListResult {
        NCListResult {
            start: entry.start,
            end: entry.end,
            id: entry.id,
            parents: entry.parents.clone(),
        }
    }
}
impl std::convert::From<NCListEntry> for NCListResult {
    fn from(entry: NCListEntry) -> NCListResult {
        NCListResult::new(&entry)
    }
}

impl std::convert::From<&NCListEntry> for NCListResult {
    fn from(entry: &NCListEntry) -> NCListResult {
        NCListResult::new(entry)
    }
}

trait ToIntervals {
    fn to_interval_vec(&self) -> Vec<Range<u32>>;
}

impl ToIntervals for Vec<NCListResult> {
    fn to_interval_vec(&self) -> Vec<Range<u32>> {
        return self.iter().map(|entry| entry.start..entry.end).collect();
    }
}
impl ToIntervals for Vec<&NCListResult> {
    fn to_interval_vec(&self) -> Vec<Range<u32>> {
        return self.iter().map(|entry| entry.start..entry.end).collect();
    }
}

#[derive(Debug)]
struct NCList {
    root: NCListEntry,
    length: usize,
}

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
    fn new(mut ranges: Vec<Range<u32>>) -> NCList {
        ranges.sort_unstable_by(nclist_range_sort);
        //now the vector contains all children, in order, right after ther parent
        //and all I need to do is reassemble them into the tree, right?
        let start = ranges[0].start;
        let stop = std::u32::MAX;
        let mut root = NCListEntry::new(start, stop, 0);
        let mut it = ranges.iter().enumerate().peekable();
        NCList::build_tree(&mut root, &mut it);
        NCList { root: root, length: ranges.len() }
    }
    /// recursivly build a nested containment list
    /// out of the sorted intervals
    fn build_tree(
        parent: &mut NCListEntry,
        it: &mut std::iter::Peekable<
            std::iter::Enumerate<std::slice::Iter<'_, std::ops::Range<u32>>>,
        >,
    ) {
        loop {
            match it.peek() {
                Some((_, next)) => {
                    if next.start > parent.end {
                        return;
                    }
                    let (ii, r) = it.next().unwrap();
                    if r.start > r.end {
                        panic!("invalid interval end < start");
                    }
                    let entry = NCListEntry::new(r.start, r.end, ii as u32);
                    parent.sublist.push(entry);
                    NCList::build_tree(parent.sublist.last_mut().unwrap(), it);
                }
                None => {
                    return;
                }
            }
        }
    }

    fn has_overlap(&self, query: Range<u32>) -> bool {
        if query.start > query.end {
            panic!("invalid interval end < start");
        }
        //has overlap is easy because all we have to do is scan the first level
        let sublist = &self.root.sublist[..];
        //find the first interval that has a stop > query.start
        //this is also the left most interval in terms of start with such a stop
        let first = sublist.upper_bound_by_key(&query.start, |entry| entry.end);
        if first == sublist.len() {
            // ie no entry larger...
            println!("none found");
            return false;
        }
        let next = &sublist[first];
        if query.start > next.end {
            return false;
        }
        if next.start >= query.end {
            return false;
        }
        return true;
        if (next.start < query.start) && (next.end > query.end) {
            return false;
        }
        return true;
    }

    fn to_vec(&self) -> Vec<NCListResult> {
        let mut collector  = Vec::new();
        self.root.collect_into_vector(&mut collector);
        return collector;
    }

    fn query_overlapping(&self, query: Range<u32>) -> Vec<NCListResult> {
        let mut res = Vec::new();
        self.root.query_overlapping(&query, &mut res);
        return res;
    }
    fn query_overlapping_multiple(&self, query: &[Range<u32>]) -> Vec<NCListResult> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        let mut keep = vec![false; self.len()];
        for q in query {
            self.root.tag_overlapping(&q, &mut keep);
        }
        let mut res = Vec::new();
        self.root.retrieve_tagged(&keep, &mut res);
        return res;
    }
    fn query_non_overlapping_multiple(&self, query: &[Range<u32>]) -> Vec<NCListResult> {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        let mut keep = vec![false; self.len()];
        for q in query {
            self.root.tag_overlapping(&q, &mut keep);
        }
        //keep = !keep;
        //todo: fix this
        let mut res = Vec::new();
        self.root.retrieve_tagged(&keep, &mut res);
        return res;
    }

    fn len(&self) -> usize {
        self.length
    }
}

struct NCListIterator<'a> {
    list: &'a NCList,
    stack: Vec<&'a NCListEntry>,
    pos_in_stack: Vec<usize>,
}


#[cfg(test)]
mod tests {
    use crate::{NCList, NCListResult, ToIntervals};
    #[test]
    fn test_has_overlap() {
        let r = vec![0..5, 10..15];
        let n = NCList::new(r);
        assert!(n.root.sublist.len() == 2);
        assert!(n.has_overlap(3..4));
        assert!(n.has_overlap(5..20));
        assert!(!n.has_overlap(6..10));
        assert!(!n.has_overlap(100..110));

        let r2 = vec![10..15, 0..5];
        let n = NCList::new(r2);
        assert!(n.root.sublist.len() == 2);

        let r2 = vec![0..15, 0..6];
        let n = NCList::new(r2);
        assert!(n.root.sublist.len() == 1);
        assert!(n.root.sublist[0].sublist.len() == 1);
        assert!(n.has_overlap(3..4));
        assert!(n.has_overlap(5..20));
        assert!(n.has_overlap(6..10));
        assert!(!n.has_overlap(20..30));

        let r2 = vec![100..150, 30..40, 200..400];
        let n = NCList::new(r2);
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
        let n = NCList::new(vec![100..150, 30..40, 200..400, 250..300]);
        assert!(n.root.sublist.len() == 3);
        let c: Vec<NCListResult> = n.to_vec();
        assert!(!c.is_empty());
        let c = c.to_interval_vec();
        dbg!(&c);
        assert!(c == vec![30..40, 100..150, 200..400, 250..300]);
    }

    #[test]
    fn test_query() {
        let n = NCList::new(vec![100..150, 30..40, 200..400, 250..300]);
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
        let n = NCList::new(vec![100..150, 30..40, 200..400, 250..300]);
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[0..5,
                                                                0..105]);
        assert!(c.to_interval_vec() == vec![30..40, 100..150]);
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[500..600,
                                                                550..700]);
        assert!(c.is_empty());
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[
                                                                45..230]);
        assert!(c.to_interval_vec() == vec![100..150, 200..400]);
        let c: Vec<NCListResult> = n.query_overlapping_multiple(&[
                                                                45..101,
                                                                101..230]);
        assert!(c.to_interval_vec() == vec![100..150, 200..400]);

    }
}
