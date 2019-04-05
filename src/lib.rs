#![feature(nll)]
use std::cmp::{max, min, Ordering};
use std::collections::HashMap;
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
pub struct NCList {
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
                    if (parent.no != -1) && (next.end > self.intervals[parent.no as usize].end) {
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
        return next.overlaps(&query);
    }

    pub fn iter(
        &self,
    ) -> std::iter::Zip<std::slice::Iter<'_, std::ops::Range<u32>>, std::slice::Iter<'_, Vec<u32>>>
    {
        self.intervals.iter().zip(self.ids.iter())
    }

    pub fn query_overlapping(&mut self, query: Range<u32>) -> NCList {
        let keep = self._tag_overlapping(&[query]);
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
            let mut it = 1..(self.len());
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

    pub fn merge_drop(&self) -> NCList {
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

    pub fn merge_split(&mut self) -> NCList {
        dbg!(&self);
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
                println!(
                    "last: {:?}, next{:?}, last_ids: {:?}",
                    &last, &next, last_ids
                );
                if (last.kind == SiteKind::Start)
                    || ((last.kind == SiteKind::End) && (next.kind == SiteKind::End))
                {
                    println!("adding {}..{}", last.pos, next.pos);
                    new_intervals.push(last.pos..next.pos);
                    let mut ids_here: Vec<u32> = last_ids.keys().map(|x| *x).collect();
                    ids_here.sort();
                    new_ids.push(ids_here);
                }
                match next.kind {
                    SiteKind::Start => {
                        for id in &next.id {
                            println!("pushing {}", *id);
                            last_ids.insert(*id, 1);
                        }
                    }
                    SiteKind::End => {
                        for id in &next.id {
                            println!("popping {}", *id);
                            *last_ids.get_mut(id).unwrap() -= 1;
                        }
                        last_ids.retain(|_k, v| (*v > 0));
                    }
                }
                last = next;
            }
        }
        NCList {
            intervals: new_intervals,
            ids: new_ids,
            root: None,
        }
        .filter_empty()
    }

    fn filter_empty(&self) -> NCList {
        let keep = self.intervals.iter().map(|r| r.start != r.end).collect();
        return self.new_filtered(&keep);
    }

    pub fn find_closest_start_left(&mut self, pos: u32) -> Option<(Range<u32>, Vec<u32>)> {
        let first = self.intervals.upper_bound_by_key(&pos, |entry| entry.start);
        if first == 0 {
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
        if first == self.len() {
            return None;
        }
        return Some((self.intervals[first].clone(), self.ids[first].clone()));
    }

    pub fn find_closest_start(&mut self, pos: u32) -> Option<(Range<u32>, Vec<u32>)> {
        let left = self.find_closest_start_left(pos);
        let right = self.find_closest_start_left(pos);
        match (left, right) {
            (None, None) => None,
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (Some(l), Some(r)) => {
                let distance_left = (l.0.start as i64 - pos as i64).abs();
                let distance_right = (r.0.start as i64 - pos as i64).abs();
                if distance_left <= distance_right {
                    Some(l)
                } else {
                    Some(r)
                }
            }
        }
    }

    pub fn covered_units(&mut self) -> u32 {
        let merged = self.merge_hull();
        let mut total = 0;
        for iv in merged.intervals.iter() {
            total += iv.end - iv.start;
        }
        return total;
    }

    pub fn mean_interval_size(&self) -> f64 {
        let mut total = 0;
        for iv in self.intervals.iter() {
            total += iv.end - iv.start;
        }
        total as f64 / self.len() as f64
    }

    pub fn invert(&self, lower_bound: u32, upper_bound: u32) -> NCList {
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
        NCList::new_presorted(new_intervals, new_ids)
    }

    pub fn filter_to_overlapping(&mut self, other: NCList) -> NCList {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        let keep = self._tag_overlapping(&other.intervals);
        self.new_filtered(&keep)
    }

    pub fn filter_to_non_overlapping(&mut self, query: &[Range<u32>]) -> NCList {
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

    pub fn filter_to_overlapping_k_others(&mut self, others: &[&NCList], min_k: u32) -> NCList {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        //  let keep = self._tag_overlapping(query);
        // self.new_filtered(&keep)

        let counts = self._count_overlapping(others);
        let mut keep = vec![false; self.len()];
        for (ii, value) in counts.iter().enumerate() {
            if *value >= min_k {
                keep[ii] = true;
            }
        }
        self.new_filtered(&keep)
    }
    pub fn filter_to_non_overlapping_k_others(&mut self, others: &[&NCList], max_k: u32) -> NCList {
        //I'm not certain this is the fastest way to do this - but it does work
        //depending on the number of queries, it might be faster
        //to do it the other way around, or do full scan of all entries here
        //(we have to visit them any how to check the keep tags)
        //  let keep = self._tag_overlapping(query);
        // self.new_filtered(&keep)
        let counts = self._count_overlapping(others);
        let mut keep = vec![false; self.len()];
        for (ii, value) in counts.iter().enumerate() {
            if *value <= max_k {
                keep[ii] = true;
            }
        }
        self.new_filtered(&keep)
    }

    pub fn union(&self, others: Vec<&NCList>) -> NCList {
        let mut new_intervals: Vec<Range<u32>> = Vec::new();
        new_intervals.extend_from_slice(&self.intervals);
        for o in others {
            new_intervals.extend_from_slice(&o.intervals);
        }
        NCList::new(&new_intervals[..])
    }
    pub fn substract(&self, other: &mut NCList) -> NCList {
        let keep = self
            .intervals
            .iter()
            .enumerate()
            .map(|(_ii, iv)| !other.has_overlap(iv))
            .collect();
        return self.new_filtered(&keep);
    }

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

    fn _count_overlapping(&mut self, others: &[&NCList]) -> Vec<u32> {
        let mut counts: Vec<u32> = vec![0; self.len()];
        for o in others {
            let keep = self._tag_overlapping(&o.intervals);
            for (ii, value) in keep.iter().enumerate() {
                if *value {
                    counts[ii] += 1;
                }
            }
        }
        counts
    }
}

impl Eq for NCList {}
impl PartialEq for NCList {
    fn eq(&self, other: &NCList) -> bool {
        (self.intervals == other.intervals) && (self.ids == other.ids)
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
        assert!(n.has_overlap(&(3..4)));
        assert!(n.has_overlap(&(5..20)));
        assert!(!n.has_overlap(&(6..10)));
        assert!(!n.has_overlap(&(100..110)));

        let r2 = vec![0..15, 0..6];
        let mut n = NCList::new(&r2);
        assert!(n.has_overlap(&(3..4)));
        assert!(n.has_overlap(&(5..20)));
        assert!(n.has_overlap(&(6..10)));
        assert!(!n.has_overlap(&(20..30)));

        let r2 = vec![100..150, 30..40, 200..400];
        let mut n = NCList::new(&r2);
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
        let c = n.query_overlapping(0..5);
        assert!(c.is_empty());
        let c = n.query_overlapping(0..31);
        assert_eq!(c.intervals, vec![30..40]);
        let c = n.query_overlapping(200..250);
        assert_eq!(c.intervals, vec![200..400]);
        let c = n.query_overlapping(200..251);
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c = n.query_overlapping(0..1000);
        dbg!(&c);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.query_overlapping(401..1000);
        assert!(c.is_empty());
    }
    #[test]
    fn test_query_multiple() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_overlapping(NCList::new(&vec![0..5, 0..105]));
        assert_eq!(c.intervals, vec![30..40, 100..150]);
        let c = n.filter_to_overlapping(NCList::new(&vec![500..600, 550..700]));
        assert!(c.is_empty());
        let c = n.filter_to_overlapping(NCList::new(&vec![45..230]));
        assert_eq!(c.intervals, vec![100..150, 200..400]);
        let c = n.filter_to_overlapping(NCList::new(&vec![45..101, 101..230]));
        assert_eq!(c.intervals, vec![100..150, 200..400]);
    }

    #[test]
    fn test_query_multiple_non_overlapping() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&[0..5, 0..105]);
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&[500..600, 550..700]);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping(&[0..600]);
        assert!(c.is_empty());
        let c = n.filter_to_non_overlapping(&[45..230]);
        assert_eq!(c.intervals, vec![30..40, 250..300]);
        let c = n.filter_to_non_overlapping(&[45..101, 101..230]);
        assert_eq!(c.intervals, vec![30..40, 250..300]);
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
        assert_eq!(n.len(), 1);

        let n = NCList::new(&vec![30..40, 30..40, 100..150]).remove_duplicates();
        assert!(!n.any_overlapping());
        assert_eq!(n.len(), 2);
        let n = NCList::new(&vec![30..40, 30..40, 35..150]).remove_duplicates();
        assert_eq!(n.len(), 2);
        let n = NCList::new_with_ids(
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
        let n = NCList::new(&vec![100..150]).merge_hull();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());

        let n = NCList::new(&vec![100..150, 120..180]).merge_hull();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![100..180]);
        assert_eq!(n.ids, vec![vec![0, 1]]);

        let n = NCList::new(&vec![100..150, 120..180, 110..115]).merge_hull();
        assert!(n.len() == 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![100..180]);
        assert_eq!(n.ids, vec![vec![0, 1, 2]]);

        let n = NCList::new(&vec![100..150, 120..180, 110..115, 200..201]).merge_hull();
        assert_eq!(n.len(), 2);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![100..180, 200..201]);
        assert_eq!(n.ids, vec![vec![0, 1, 2], vec![3]]);
    }

    #[test]
    fn test_merge_drop() {
        let n = NCList::new(&vec![]).merge_drop();
        assert_eq!(n.len(), 0);
        assert!(!n.any_overlapping());

        let n = NCList::new(&vec![100..150]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());

        let n = NCList::new(&vec![100..150, 120..180]).merge_drop();
        assert_eq!(n.len(), 0);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![]);
        assert_eq!(n.ids, Vec::<Vec<u32>>::new());

        let n = NCList::new(&vec![100..150, 120..180, 200..250]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![2]]);

        let n = NCList::new(&vec![100..150, 120..180, 200..250, 106..110]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![3]]);

        let n = NCList::new(&vec![100..150, 120..180, 200..250, 106..110, 80..105]).merge_drop();
        assert_eq!(n.len(), 1);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![200..250]);
        assert_eq!(n.ids, vec![vec![4]]);

        let n = NCList::new(&vec![
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
        assert_eq!(n.len(), 3);
        assert!(!n.any_overlapping());
        assert_eq!(n.intervals, vec![30..40, 200..250, 400..405]);
        assert_eq!(n.ids, vec![vec![0], vec![5], vec![6]]);
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
        let mut n = NCList::new(&vec![]);
        assert!(n.find_closest_start_right(29).is_none());
    }
    fn test_find_closest_start() {
        let mut n = NCList::new(&vec![]);
        assert!(n.find_closest_start(100).is_none());
        let mut n = NCList::new(&vec![100..110, 200..300]);
        assert_eq!(n.find_closest_start(0).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(100).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(149).unwrap(), (100..110, vec![0]));
        assert_eq!(n.find_closest_start(150).unwrap(), (200..300, vec![1]));
        assert_eq!(n.find_closest_start(151).unwrap(), (200..300, vec![1]));
        assert_eq!(n.find_closest_start(251).unwrap(), (200..300, vec![1]));
        assert_eq!(n.find_closest_start(351).unwrap(), (200..300, vec![1]));
    }

    #[test]
    fn test_covered_units() {
        let mut n = NCList::new(&vec![]);
        assert_eq!(n.covered_units(), 0);
        let mut n = NCList::new(&vec![10..100]);
        assert_eq!(n.covered_units(), 90);
        let mut n = NCList::new(&vec![10..100, 200..300]);
        assert_eq!(n.covered_units(), 90 + 100);
        let mut n = NCList::new(&vec![10..100, 200..300, 15..99]);
        assert_eq!(n.covered_units(), 90 + 100);
        let mut n = NCList::new(&vec![10..100, 200..300, 15..99, 15..105]);
        assert_eq!(n.covered_units(), 90 + 100 + 5);
    }

    #[test]
    fn test_mean_interval_size() {
        let n = NCList::new(&vec![]);
        assert!(n.mean_interval_size().is_nan());
        let n = NCList::new(&vec![10..100]);
        assert_eq!(n.mean_interval_size(), 90.);
        let n = NCList::new(&vec![10..100, 200..300]);
        assert_eq!(n.mean_interval_size(), (90 + 100) as f64 / 2.0);
        let n = NCList::new(&vec![10..100, 200..300, 15..99]);
        assert_eq!(n.mean_interval_size(), (90 + 100 + (99 - 15)) as f64 / 3.0);
        let n = NCList::new(&vec![10..100, 200..300, 15..99, 15..105]);
        assert_eq!(
            n.mean_interval_size(),
            (((100 - 10) + (300 - 200) + (99 - 15) + (105 - 15)) as f64 / 4.0)
        );
    }

    #[test]
    fn test_invert() {
        let n = NCList::new(&vec![]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..100,]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = NCList::new(&vec![30..40]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 40..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = NCList::new(&vec![30..40, 35..38]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 40..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = NCList::new(&vec![30..40, 35..38, 35..50]).invert(0, 100);
        assert_eq!(n.intervals, vec![0..30, 50..100,]);
        assert_eq!(n.ids, vec![vec![0], vec![1]]);
        let n = NCList::new(&vec![30..40, 35..38, 35..50]).invert(40, 100);
        assert_eq!(n.intervals, vec![50..100,]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = NCList::new(&vec![30..40, 35..38, 35..50, 55..60]).invert(40, 40);
        assert_eq!(n.intervals, vec![50..55]);
        assert_eq!(n.ids, vec![vec![0]]);
        let n = NCList::new(&vec![30..40, 35..38, 35..50]).invert(40, 40);
        assert!(n.intervals.is_empty());
        assert!(n.ids.is_empty());
    }

    #[test]
    fn test_union() {
        let n = NCList::new(&vec![]).union(vec![&NCList::new(&vec![0..100])]);
        assert_eq!(n.intervals, vec![0..100]);

        let n = NCList::new(&vec![0..10]).union(vec![&NCList::new(&vec![0..100])]);
        assert_eq!(n.intervals, vec![0..100, 0..10]);

        let n = NCList::new(&vec![0..10]).union(vec![&NCList::new(&vec![0..100, 200..300])]);
        assert_eq!(n.intervals, vec![0..100, 0..10, 200..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![2]]);

        let n = NCList::new(&vec![0..10]).union(vec![&NCList::new(&vec![])]);
        assert_eq!(n.intervals, vec![0..10]);
        let n = NCList::new(&vec![0..10]).union(vec![
            &NCList::new(&vec![0..100]),
            &NCList::new(&vec![200..300]),
        ]);
        assert_eq!(n.intervals, vec![0..100, 0..10, 200..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn test_substract() {
        let n = NCList::new(&vec![]).substract(&mut NCList::new(&vec![0..100]));
        assert!(n.intervals.is_empty());

        let n = NCList::new(&vec![0..10]).substract(&mut NCList::new(&vec![0..100]));
        assert!(n.intervals.is_empty());

        let n = NCList::new(&vec![0..10, 100..150]).substract(&mut NCList::new(&vec![0..100]));
        assert_eq!(n.intervals, vec![100..150]);

        let n = NCList::new(&vec![0..10, 100..150, 150..300])
            .substract(&mut NCList::new(&vec![55..101]));
        assert_eq!(n.intervals, vec![0..10, 150..300]);
        assert_eq!(n.ids, vec![vec![0], vec![2]]);

        let n = NCList::new(&vec![0..10, 5..6, 100..150, 150..300])
            .substract(&mut NCList::new(&vec![55..101]));
        assert_eq!(n.intervals, vec![0..10, 5..6, 150..300]);
        assert_eq!(n.ids, vec![vec![0], vec![1], vec![3]]);
    }

    #[test]
    fn test_filter_overlapping_multiples() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_overlapping_k_others(&[&NCList::new(&vec![0..5, 0..105])], 1);
        assert_eq!(c.intervals, vec![30..40, 100..150]);
        let c = n.filter_to_overlapping_k_others(&[&NCList::new(&vec![0..5, 0..105])], 0);
        assert_eq!(c, n);
        let c = n.filter_to_overlapping_k_others(&[&NCList::new(&vec![0..5, 0..105])], 2);
        assert!(c.is_empty());

        let c = n.filter_to_overlapping_k_others(
            &[&NCList::new(&vec![0..35]), &NCList::new(&vec![0..160])],
            2,
        );
        assert_eq!(c.intervals, vec![30..40,]);
        let c = n.filter_to_overlapping_k_others(
            &[&NCList::new(&vec![0..35]), &NCList::new(&vec![0..160])],
            1,
        );
        assert_eq!(c.intervals, vec![30..40, 100..150]);
    }

    #[test]
    fn test_filter_non_overlapping_multiples() {
        let mut n = NCList::new(&vec![100..150, 30..40, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(&[&NCList::new(&vec![0..5, 0..105])], 1);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(&[&NCList::new(&vec![0..5, 0..105])], 0);
        assert_eq!(c.intervals, vec![200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(&[&NCList::new(&vec![0..5, 0..105])], 2);
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);

        let c = n.filter_to_non_overlapping_k_others(
            &[&NCList::new(&vec![0..35]), &NCList::new(&vec![0..160])],
            2,
        );
        assert_eq!(c.intervals, vec![30..40, 100..150, 200..400, 250..300]);
        let c = n.filter_to_non_overlapping_k_others(
            &[&NCList::new(&vec![0..35]), &NCList::new(&vec![0..160])],
            1,
        );
        assert_eq!(c.intervals, vec![100..150, 200..400, 250..300]);
    }

    #[test]
    fn test_split() {
        let mut n = NCList::new(&vec![0..100, 20..30]);
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..20, 20..30, 30..100]);
        assert_eq!(c.ids, vec![vec![0], vec![0, 1,], vec![0]]);
        println!("");

        let mut n = NCList::new(&vec![0..100, 0..90, 70..95, 110..150]);
        let c = n.merge_split();
        assert_eq!(c.intervals, [0..70, 70..90, 90..95, 95..100, 110..150]);
        assert_eq!(
            c.ids,
            vec![vec![0, 1], vec![0, 1, 2], vec![0, 2], vec![0], vec![3]]
        );
        let mut n = NCList::new_with_ids(
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

}
