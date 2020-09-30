#!/usr/bin/env python3

# Doubly Linked List Node
class LinkedListNode(object):
    def __init__(self, x):
        self.obj = x
        self.prev = None
        self.next = None
    
    def __call__(self):
        return self.obj

    def get_data(self):
        return self.__call__()

# Doubly Linked List
class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_before(self, item:LinkedListNode, x:LinkedListNode):
        x.next = item
        x.prev = item.prev
        if (None == item.prev):
            x.next = item
            item.prev = x
            self.head = x
        else:
            item.prev.next = x
            item.prev = x
        return

    def insert_after(self, item:LinkedListNode, x:LinkedListNode):
        x.next = item.next
        x.prev = item
        if (None == item.next):
            item.next = x
            x.prev = item
            self.tail = x
        else:
            item.next.prev = x
            item.next = x
        return

    def insert_head(self, x: LinkedListNode):
        if (self.empty()):
            assert(None == self.tail)
            self.head = self.tail = item
            x.next = x.prev = None
        else:
            self.insert_before(self.head, x)
    
    def insert_tail(self, x: LinkedListNode):
        if (self.empty()):
            assert(None == self.head)
            self.head = self.tail = item 
            x.next = x.prev = None
        else:
            self.insert_after(self.tail, x)

    def insert_head(self, item: LinkedListNode):
        if (None == self.head):
            assert(None == self.tail)
            self.head = self.tail = item 
        else:
            self.insert_before(self.head, item)

    def items(self):
        current = self.head
        while (None != current):
            yield current.obj
            current = current.next

    def nodes(self):
        current = self.head
        while (None != current):
            yield current
            current = current.next
    
    def find_node(self, x) -> LinkedListNode:
        for node in self.nodes():
            if node.obj == x:
                return node

    def delete_node(self, node:LinkedListNode):
        if (None == node):
            return
        if (self.head != node and self.tail != node):
            assert(None != node.next and None != node.prev)
            node.prev.next = node.next
            node.next.prev = node.prev
        else:
            if (self.head == node):
                self.head = node.next
            if (self.tail == node):
                self.tail = node.prev
        del node

    def front(self) -> LinkedListNode:
        return self.head

    def back(self) -> LinkedListNode:
        return self.tail

    def empty(self):
        ret = (None == self.head)
        if (True == ret):
            assert(None == self.tail)
        return ret


"""
#Some tests
ll = LinkedList()
ll.insert_head(LinkedListNode("one"))
ll.insert_head(LinkedListNode("two"))
ll.insert_head(LinkedListNode("three"))
ll.insert_head(LinkedListNode("four"))
ll.insert_tail(LinkedListNode("zero"))
node = ll.find_node("three")
ll.insert_before(node, LinkedListNode("three.five"))
ll.insert_after(node, LinkedListNode("two.five"))
for i in ll.nodes():
    print(i())
"""