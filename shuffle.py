import torch

def shuffle_data(data, seed=0):
	images, captions = data
	shuffled_images = []
	shuffled_captions = []
	num_images = len(images)
	torch.manual_seed(seed)
	perm = list(torch.randperm(num_images))
	for i in range(num_images):
		shuffled_images.append(images[perm[i]])
		shuffled_captions.append(captions[perm[i]])
	return shuffled_images, shuffled_captions