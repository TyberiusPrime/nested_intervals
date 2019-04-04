#![feature(nll)]
use std::cmp::Ordering;
use std::ops::Range;
use superslice::*;

trait FilterByBools<T> {
    fn filter_by_bools(&self, keep: &Vec<bool>) -> Vec<T>;
}

impl<T> FilterByBools<T> for Vec<T>
where
    T: Clone,
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
    ids: Vec<Vec<u32>>,
    root: Option<NCListEntry>,
}

#[derive(Debug)]
pub struct NCListEntry {
    no: i32,
    children: Vec<NCListEntry>,
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
            ids: (0..count).map(|x| vec![x as u32]).collect(),
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
        let ids = ids.iter().map(|i| vec![*i]).collect();
        NCList {
            intervals: iv,
            ids: ids,
            root: None,
        }
    }
    fn new_filtered(&self, keep: &Vec<bool>) -> NCList {
        NCList {
            intervals: self.intervals.filter_by_bools(&keep),
            ids: self.ids.filter_by_bools(&keep),
            root: None,
        }
    }

    /// used by the merge functions to bypass the sorting and checking
    /// on already sorted & checked intervals
    fn new_presorted(intervals: Vec<Range<u32>>, ids: Vec<Vec<u32>>) -> NCList {
        NCList {
            intervals: intervals,
            ids: ids,
            root: None,
        }
    }

    fn len(&self) -> usize {
        self.intervals.len()
    }

    fn is_empty(&self) -> bool {
        return self.len() == 0;
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
    ) -> std::iter::Zip<std::slice::Iter<'_, std::ops::Range<u32>>, std::slice::Iter<'_, Vec<u32>>>
    {
        self.intervals.iter().zip(self.ids.iter())
    }

    pub fn filter_to_overlapping(&mut self, query: Range<u32>) -> NCList {
        return self.filter_to_overlapping_multiple(&[query]);
    }

    pub fn filter_to_overlapping_multiple(&mut self, query: &[Range<u32>]) -> NCList {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        let keep = self._tag_overlapping(query);
        self.new_filtered(&keep)
    }

    pub fn filter_to_non_overlapping_multiple(&mut self, query: &[Range<u32>]) -> NCList {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        let mut keep = self._tag_overlapping(query);
        for b in keep.iter_mut() {
            *b = !*b;
        }
        self.new_filtered(&keep)
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
        self.new_filtered(&keep)
    }

    pub fn merge_hull(&self) -> NCList {
        let mut new_intervals: Vec<Range<u32>> = Vec::new();
        let mut new_ids: Vec<Vec<u32>> = Vec::new();
        if !self.is_empty() {
            let mut last = self.intervals[0].clone();
            let mut last_ids: Vec<u32> = self.ids[0].clone();
            let mut it = (1..(self.len()));
            loop {
                let mut ii = match it.next() {
                    Some(ii) => ii,
                    None => {
                        break;
                    }
                };
                let mut next = &self.intervals[ii];
                while last.overlaps(next) {
                    if next.end > last.end {
                        // no point in extending internal intervals
                        last.end = next.end;
                    }
                    last_ids.extend_from_slice(&self.ids[ii]);
                    ii = match it.next() {
                        Some(ii) => ii,
                        None => {
                            break;
                        }
                    };
                    next = &self.intervals[ii];
                }
                new_intervals.push(last);
                new_ids.push(last_ids);
                last = next.clone();
                last_ids = self.ids[ii].clone();
            }
            if new_intervals.is_empty() || (new_intervals.last().unwrap().end < last.start) {
                new_intervals.push(last);
                new_ids.push(last_ids);
            }
        }
        NCList::new_presorted(new_intervals, new_ids)
    }

    pub fn merge_drop(&mut self) -> NCList {
        let mut keep = vec![true; self.len()];
        let mut last_stop = 0;
        self.ensure_nclist();
        for ii in (0..self.len()) {
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

    pub fn find_closest_start_left(&mut self, pos: u32) -> Option<(Range<u32>, Vec<u32>)> {
        let first = self.intervals.upper_bound_by_key(&pos, |entry| entry.start);
        if (first == 0) {
            return None;
        }
        let prev = first - 1;
        return Some((self.intervals[prev].clone(), self.ids[prev].clone()));
    }

    pub fn find_closest_start_right(&mut self, pos: u32) -> Option<(Range<u32>, Vec<u32>)> {
        let first = self
            .intervals
            .upper_bound_by_key(&pos, |entry| entry.start + 1);
        // since this the first element strictly greater, we have to do -1
        if (first == self.len()) {
            return None;
        }
        return Some((self.intervals[first].clone(), self.ids[first].clone()));
    }

    fn _tag_overlapping_recursion(
        &self,
        node: &NCListEntry,
        query: &Range<u32>,
        tags: &mut Vec<bool>,
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
            if !next_iv.overlaps(query) {
                return;
            }
            tags[next.no as usize] = true;
            if !next.children.is_empty() {
                self._tag_overlapping_recursion(next, query, tags);
            }
        }
    }

    fn _tag_overlapping(&mut self, query: &[Range<u32>]) -> Vec<bool> {
        self.ensure_nclist();
        let mut keep = vec![false; self.len()];
        for q in query {
            self._tag_overlapping_recursion(&self.root.as_ref().unwrap(), &q, &mut keep);
        }
        keep
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
    use crate::NCList;
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
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_overlapping(0..5);
        assert!(c.is_empty());
        let c = n.filter_to_overlapping(0..31);
        assert!(c.intervals == vec![30..40]);
        let c = n.filter_to_overlapping(200..250);
        assert!(c.intervals == vec![200..400]);
        let c = n.filter_to_overlapping(200..251);
        assert!(c.intervals == vec![200..400, 250..300]);
        let c = n.filter_to_overlapping(0..1000);
        dbg!(&c);
        assert!(c.intervals == vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_overlapping(401..1000);
        assert!(c.is_empty());
    }
    #[test]
    fn test_query_multiple() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_overlapping_multiple(&[0..5, 0..105]);
        assert!(c.intervals == vec![30..40, 100..150]);
        let c = n.filter_to_overlapping_multiple(&[500..600, 550..700]);
        assert!(c.is_empty());
        let c = n.filter_to_overlapping_multiple(&[45..230]);
        assert!(c.intervals == vec![100..150, 200..400]);
        let c = n.filter_to_overlapping_multiple(&[45..101, 101..230]);
        assert!(c.intervals == vec![100..150, 200..400]);
    }
    #[test]
    fn test_query_multiple_non_overlapping() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_multiple(&[0..5, 0..105]);
        assert!(c.intervals == vec![200..400, 250..300]);
        let c = n.filter_to_non_overlapping_multiple(&[500..600, 550..700]);
        assert!(c.intervals == vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_multiple(&[0..600]);
        assert!(c.is_empty());
        let c = n.filter_to_non_overlapping_multiple(&[45..230]);
        assert!(c.intervals == vec![30..40, 250..300]);
        let c = n.filter_to_non_overlapping_multiple(&[45..101, 101..230]);
        assert!(c.intervals == vec![30..40, 250..300]);
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
            &[55, 56, 57, 58, 59],
        )
        .remove_duplicates();
        assert!(n.len() == 3);
        dbg!(&n);
        assert!(n.ids == vec![vec![55], vec![57], vec![59]]);
    }

    #[test]
    fn test_merge_hull() {
        let n = NCList::new(&vec![100..150]).merge_hull();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());

        let n = NCList::new(&vec![100..150, 120..180]).merge_hull();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![100..180]);
        assert!(n.ids == vec![vec![0, 1]]);

        let n = NCList::new(&vec![100..150, 120..180, 110..115]).merge_hull();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![100..180]);
        assert!(n.ids == vec![vec![0, 1, 2]]);

        let n = NCList::new(&vec![100..150, 120..180, 110..115, 200..201]).merge_hull();
        assert!(n.len() == 2);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![100..180, 200..201]);
        assert!(n.ids == vec![vec![0, 1, 2], vec![3]]);
    }

    #[test]
    fn test_merge_drop() {
        let n = NCList::new(&vec![]).merge_drop();
        assert!(n.len() == 0);
        assert!(!n.any_overlapping());

        let n = NCList::new(&vec![100..150]).merge_drop();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());

        let n = NCList::new(&vec![100..150, 120..180]).merge_drop();
        assert!(n.len() == 0);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![]);
        assert!(n.ids == Vec::<Vec<u32>>::new());

        let n = NCList::new(&vec![100..150, 120..180, 200..250]).merge_drop();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![200..250]);
        assert!(n.ids == vec![vec![2]]);

        let n = NCList::new(&vec![100..150, 120..180, 200..250, 106..110]).merge_drop();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![200..250]);
        assert!(n.ids == vec![vec![3]]);

        let n = NCList::new(&vec![100..150, 120..180, 200..250, 106..110, 80..105]).merge_drop();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![200..250]);
        assert!(n.ids == vec![vec![4]]);

        let n = NCList::new(&vec![
            100..150,
            120..180,
            200..250,
            106..110,
            80..105,
            30..40,
        ])
        .merge_drop();
        assert!(n.len() == 2);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![30..40, 200..250]);
        assert!(n.ids == vec![vec![0], vec![5]]);

        let n = NCList::new(&vec![
            100..150,
            120..180,
            200..250,
            106..110,
            80..105,
            30..40,
            400..405,
        ])
        .merge_drop();
        assert!(n.len() == 3);
        assert!(!n.any_overlapping());
        assert!(n.intervals == vec![30..40, 200..250, 400..405]);
        assert!(n.ids == vec![vec![0], vec![5], vec![6]]);
    }

    #[test]
    fn test_find_closest_start_left() {
        let mut n = NCList::new(&vec![
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
        assert!(n.find_closest_start_left(100).unwrap() == (100..150, vec![2]));
        assert!(n.find_closest_start_left(105).unwrap() == (100..150, vec![2]));
        assert!(n.find_closest_start_left(106).unwrap() == (106..110, vec![4]));
        assert!(n.find_closest_start_left(109).unwrap() == (107..125, vec![5]));
        assert!(n.find_closest_start_left(110).unwrap() == (107..125, vec![5]));
        assert!(n.find_closest_start_left(111).unwrap() == (107..125, vec![5]));
        assert!(n.find_closest_start_left(120).unwrap() == (120..180, vec![6]));
        assert!(n.find_closest_start_left(121).unwrap() == (120..180, vec![6]));
        assert!(n.find_closest_start_left(125).unwrap() == (120..180, vec![6]));
        assert!(n.find_closest_start_left(127).unwrap() == (120..180, vec![6]));
        assert!(n.find_closest_start_left(121000).unwrap() == (400..405, vec![8]));
        let mut n = NCList::new(&vec![]);
        assert!(n.find_closest_start_left(29).is_none());
    }

    #[test]
    fn test_find_closest_start_right() {
        let mut n = NCList::new(&vec![
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
        assert!(n.find_closest_start_right(10).unwrap() == (30..40, vec![0]));
        assert!(n.find_closest_start_right(29).unwrap() == (30..40, vec![0]));
        assert!(n.find_closest_start_right(30).unwrap() == (30..40, vec![0]));
        assert!(n.find_closest_start_right(31).unwrap() == (80..105, vec![1]));
        assert!(n.find_closest_start_right(99).unwrap() == (100..150, vec![2]));
        assert!(n.find_closest_start_right(100).unwrap() == (100..150, vec![2]));
        assert!(n.find_closest_start_right(101).unwrap() == (106..120, vec![3]));
        assert!(n.find_closest_start_right(107).unwrap() == (107..125, vec![5]));
        assert!(n.find_closest_start_right(110).unwrap() == (120..180, vec![6]));
        assert!(n.find_closest_start_right(111).unwrap() == (120..180, vec![6]));
        assert!(n.find_closest_start_right(120).unwrap() == (120..180, vec![6]));
        assert!(n.find_closest_start_right(121).unwrap() == (200..250, vec![7]));
        assert!(n.find_closest_start_right(125).unwrap() == (200..250, vec![7]));
        assert!(n.find_closest_start_right(127).unwrap() == (200..250, vec![7]));
        assert!(n.find_closest_start_right(121000).is_none());
        let mut n = NCList::new(&vec![]);
        assert!(n.find_closest_start_right(29).is_none());
    }
}
