use std::collections::HashMap;
use std::hash::Hash;
#[derive(Debug)]
pub struct OrderedDict<K, V>{
    index: HashMap<K, usize>,
    items: Vec<Item<K,V>>,
    pub key_description: String
}

impl<K, V> OrderedDict<K,V> where K: Eq+Hash+Clone{
    pub fn new_with_key_description(description: String)->Self{
        Self{
            key_description: description,
            index: HashMap::new(),
            items: vec![]
        }
    }

    pub fn insert(&mut self, key: K, value: V)->Option<&V>{
        assert_eq!(self.index.contains_key(&key), false);
        self.index.insert(key.clone(), self.size() - 1);
        self.items.push(Item::new(key, value));
        self.items.last().map_or(None, |x|Some(&x.pair.1))
    }

    pub fn size(&self)->usize{
        self.items.len()
    }

    pub fn find(&self, key: &K)->Option<&V>{
        let it = self.index.get(&key);
        if let Some(idx)= it{
            Some(self.items[*idx].value())
        }else{
            None
        }
    }
}

#[derive(Debug)]
struct Item<K,V>{
    pair: (K,V)
}

impl<K,V> Item<K,V>{
    pub fn new(key:K, value:V)->Self{
        Self{
            pair: (key,value)
        }
        
    }

    pub fn pair(&self)->&(K,V){
        todo!()
    }

    pub fn value(&self)->&V {
        &self.pair.1
    }
    
    pub fn key(&self)->&K {
        &self.pair.0
    }

}

// impl<K, V> IntoIterator for OrderdDict<K,V> {
//     type Item = (K,V);
//     type IntoIter = IntoIter<K,V>;
//     fn into_iter(self) -> Self::IntoIter {
//         IntoIter{ base: self.base.into_iter()}
//     }

// }