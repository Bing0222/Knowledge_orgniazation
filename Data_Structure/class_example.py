# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:29:51 2022

@author: WenBi
"""

'''
Write a class(Bag),it has some functions like add,remove,len,iter

Write a test

'''

class Bag(object):
    
    def __init__(self, maxsize=10):
        self.maxisize = maxsize
        self._items = list()
        
    def add(self, item):
        if len(self)>self.maxisize:
            raise Exception('Bag is full')
        self._items.append(item)
        
    def remove(self,item):
        self._items.remove(item)
        
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        for item in self._items:
            yield item
            

def test_bag():
    bag = Bag()
    
    bag.add(1)
    bag.add(2)
    bag.add(3)
    
    assert len(bag) == 3
    
    bag.remove(3)
    assert len(bag) == 2
    
    for i in bag:
        print(i)
    
test_bag()
















            