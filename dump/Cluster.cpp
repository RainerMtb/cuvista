class Cluster {

private:
	size_t numel = 0;
	double minVal, maxVal, bucketSize;
	std::vector<std::vector<PointContext>> buckets;

public:
	Cluster(double minVal, double maxVal, int bucketCount) {
		this->minVal = minVal;
		this->maxVal = maxVal;
		this->bucketSize = (maxVal - minVal) / bucketCount;
		this->buckets.resize(bucketCount);
	}

	void add(const PointContext& pc, double value) {
		double bucketIndex = (value - minVal) / bucketSize;
		int idx = (int) bucketIndex;
		if (idx == bucketIndex && bucketIndex > 0) idx--;
		buckets[idx].push_back(pc);
		numel++;
	}

	void sortBySize() {
		std::sort(buckets.begin(), buckets.end(), [] (std::vector<PointContext>& v1, std::vector<PointContext>& v2) { return v1.size() > v2.size(); });
	}

	template <class T> T getTopPoints(double percentage, T iter) {
		size_t requiredCount = (size_t) (numel * percentage);
		size_t currentCount = 0;

		for (auto it = buckets.begin(); it != buckets.end() && currentCount < requiredCount; it++) {
			size_t n = it->size();
			iter = std::copy_n(it->begin(), n, iter);
			currentCount += n;
		}
		return iter;
	}

	friend std::ostream& operator << (std::ostream& os, const Cluster& cluster) {
		auto fcnSize = [&] (const std::vector<PointContext>& v1, const std::vector<PointContext>& v2) { return v1.size() < v2.size(); };
		size_t maxSize = std::max_element(cluster.buckets.begin(), cluster.buckets.end(), fcnSize)->size();
		size_t maxNumDots = 75;
		double midPoint = cluster.minVal + cluster.bucketSize / 2.0;
		for (size_t i = 0; i < cluster.buckets.size(); i++) {
			size_t numDots = cluster.buckets[i].size() * maxNumDots / maxSize;
			os << std::setprecision(6) << std::setw(10) << midPoint << " " << std::string(numDots, '*') << std::endl;
			midPoint += cluster.bucketSize;
		}
		return os;
	}
};

