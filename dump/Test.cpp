void fcm(const std::string& filename) {
	Matd X = Matd::fromTextFile("f:/data.txt");
	size_t numValues = X.rows();
	size_t numClusters = 6;
	double jold = 0.0;
	double jnew = 1.0;
	Matd sum = Matd::zeros(4, numClusters);
	Matd C = Matd::zeros(2, numClusters);

	auto sqr = [] (double d) { return d * d; };

	//init U such that row sums == 1.0
	Matd U = Matd::rand(numValues, numClusters, 0.0, 1.0);
	for (size_t i = 0; i < numValues; i++) {
		double sum = 0.0;
		for (size_t k = 0; k < numClusters; k++) sum += U[i][k];
		for (size_t k = 0; k < numClusters; k++) U[i][k] /= sum;
	}

	for (int iter = 0; iter < 100 && std::abs(jnew - jold) > 1e-5; iter++) {
		jold = jnew;
		jnew = 0.0;
		sum.setValues(0.0);
		C.setValues(0.0);

		//update centers
		for (size_t i = 0; i < numValues; i++) {
			for (size_t k = 0; k < numClusters; k++) {
				double uu = sqr(U[i][k]);
				sum[0][k] += uu * X[i][0];
				sum[1][k] += uu;
				sum[2][k] += uu * X[i][1];
				sum[3][k] += uu;
			}
		}
		for (size_t k = 0; k < numClusters; k++) {
			C[0][k] = sum[0][k] / sum[1][k];
			C[1][k] = sum[2][k] / sum[3][k];
		}

		//update U
		for (size_t i = 0; i < numValues; i++) {
			for (size_t k = 0; k < numClusters; k++) {
				double u = 0.0;
				double uk = sqr(X[i][0] - C[0][k]) + sqr(X[i][1] - C[1][k]);
				for (size_t j = 0; j < numClusters; j++) {
					double uj = sqr(X[i][0] - C[0][j]) + sqr(X[i][1] - C[1][j]);
					u += uk / uj;
				}
				U[i][k] = 1.0 / u;
				jnew += sqr(U[i][k]) * uk;
			}
		}

		//check sum(U) == 1
		//for (size_t i = 0; i < numValues; i++) {
		//	double sum = 0.0;
		//	for (size_t k = 0; k < numClusters; k++) sum += U[i][k];
		//	assert(std::abs(sum - 1.0) < 1e-6 && "sum U");
		//}

		//U.subMat(0, 0, 20, 6).toConsole();
		//C.toConsole();
		//std::cout << iter << ": " << jnew << std::endl;
	}

	//draw image
	int ih = 500;
	int iw = 800;
	ImageBGR image(ih, iw);
	std::vector<im::ColorBgr> colors = { im::ColorBgr::BLUE, im::ColorBgr::GREEN, im::ColorBgr::RED,
		im::ColorBgr::MAGENTA, im::ColorBgr::CYAN, im::ColorBgr::YELLOW };

	image.setValues({ 220, 220, 240 });
	for (size_t i = 0; i < numValues; i++) {
		auto it = std::max_element(U.addr(i, 0), U.addr(i, numClusters));
		if (*it > 0.5) {
			int64_t idx = std::distance(U.addr(i, 0), it);
			image.drawMarker(X[i][0] * iw, X[i][1] * ih, colors[idx], 2.0);
		}
	}
	for (size_t k = 0; k < numClusters; k++) {
		image.drawCircle(C[0][k] * iw, C[1][k] * ih, 5.0, colors[k]);
	}


	image.saveAsColorBMP(filename);
}
