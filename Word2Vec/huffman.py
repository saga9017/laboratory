import heapq
import os

class HeapNode:
	def __init__(self, char, freq):
		self.char = char
		self.freq = freq
		self.left = None
		self.right = None
		self.index=None

	def __lt__(self, other):
		if(other == None):
			return -1
		if(not isinstance(other, HeapNode)):
			return -1
		return self.freq < other.freq


class HuffmanCoding:
	def __init__(self):
		self.heap = []
		self.codes = {}
		self.reverse_mapping = {}
		self.merged_nodes=None

	def make_heap(self, frequency):
		for key in frequency:
			node = HeapNode(key, frequency[key])
			heapq.heappush(self.heap, node)

	def merge_nodes(self):
		index=0
		while(len(self.heap)>1):
			node1 = heapq.heappop(self.heap)
			node2 = heapq.heappop(self.heap)

			merged = HeapNode(None, node1.freq + node2.freq)
			merged.left = node1
			merged.right = node2
			merged.index = index
			heapq.heappush(self.heap, merged)

			index+=1

		return merged

	def make_codes_helper(self, root, current_code):
		if(root == None):
			return

		if(root.char != None):
			self.codes[root.char] = current_code
			self.reve_mapping[current_code] = root.char
			return

		self.make_codes_helper(root.left, current_code + "0")
		self.make_codes_helper(root.right, current_code + "1")


	def make_codes(self):
		root = heapq.heappop(self.heap)
		current_code = ""
		self.make_codes_helper(root, current_code)


	def build(self, frequency):
		self.make_heap(frequency)
		merged=self.merge_nodes()
		self.make_codes()

		return self.codes, merged
