sr = 32000
people = ['william', 'aaditya']#, 'lauren']
sentences = {'1': 'a quick brown fox jumps over the lazy dog and then he decided to go to the park and fool around',
				'2': 'The world is currently in chaos and my dog and I are in the safety of my warm and cozy home',
				'3': 'tomato pasta sauce with newly made meatballs and fresh picked basil and chewy mozzarella',
				'4': 'netflix is a great source of entertainment when it comes to wasting many numerous hours on end',
				'5': 'hello our names are aaditya and william and lauren and we are currently working on a security project'}
counts = {}
for si in sentences:
	counts[si] = len(sentences[si])+len(set(sentences[si])-set(sentences[si].lower())) #proxy for # of shifts
