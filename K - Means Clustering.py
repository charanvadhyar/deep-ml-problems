import math

def argmin(values):
	smallest_value = values[0]
	smallest_index = 0;

	for i in range(1, len(values)):

		if values[i] <smallest_value:
			smallest_value = values[i]
			smallest_index = i
	return smallest_index

def euclidean_distance(p1,p2):
	return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

def assign_to_clusters(points,centroids,k):

	clusters = [[] for _ in range(k)]

	

	for point in points :
		dists = [euclidean_distance(point,c) for c in centroids ]


		nearest = argmin(dists)

		clusters[nearest].append(point)
	
	return clusters


def compute_mean(cluster):
	if len(cluster) == 0:
		return None
	N =len(cluster)
	D = len(cluster[0])

	total = [0.0] * D

	for point in cluster:
		for d in range(D):
			total[d] += point[d]

	mean =[total[d]/N for d in range(D)]
	return tuple(mean)

		


def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:

	centroids = initial_centroids

	for i in range(max_iterations):
		clusters  = assign_to_clusters(points,centroids,k)

		new_centroids = [compute_mean(cluster) for cluster in clusters]
        # Handle empty clusters: fall back to old position
		centroids = [new_centroids[j] if new_centroids[j] is not None else centroids[j] for j in range(k)]
	return centroids
