// hello.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

class Distance {
public:
	int tri;
	int tci;
	double weight;
	Distance() {
		weight = 0;
	};

	Distance(int tri, int tci, double weight) : tri(tri), tci(tci),
		weight(weight) {}

	bool isToSink() {
		if (tri == -2 && tci == -2)
			return true;
		return false;
	}
	bool isToSource() {
		if (tri == -1 && tci == -1)
			return true;
		return false;
	}
};

class VertexPi {
public:
	int ri;
	int ci;
	int pri;
	int pci;
	bool isTraversed;
	vector< Distance > dList;
	VertexPi() {
		ri = 0;
		ci = 0;
		isTraversed = false;
	};
	VertexPi(int ri, int ci) : ri(ci), ci(ci) {}

	void setParent(int ri, int ci) {
		pri = ri;
		pci = ci;
	};
	void addd(int tri, int tci, float weight) {
		dList.push_back(Distance(tri, tci, weight));
	};

	Distance & getd(int ri, int ci) {
		for (int i = 0; i < dList.size(); i++) {
			Distance d = dList.at(i);
			if (d.tri == ri && d.tci == ci)
				return dList.at(i);
		}
	};

	bool isSink() {
		if (ri == -2 && ci == -2)
			return true;
		return false;
	}
	bool isSource() {
		if (ri == -1 && ci == -1)
			return true;
		return false;
	}
};

class DotI {
public:
	int a;
	int b;
	DotI(int indexa, int indexb) {
		a = indexa;
		b = indexb;
	}
};


float fordFulkerson(vector< vector< VertexPi > > &aLi, VertexPi &superSource, VertexPi &superSink, int rows, int cols, Mat &out_image);
bool search(VertexPi pixel, vector< vector< VertexPi > > &aLi, VertexPi &siNode);
double getweight(Distance d, double maxdVal);

int main(int argc, char **argv)
{
	namedWindow("Original image", WINDOW_AUTOSIZE);
	if (argc != 4) {
		cout << "Usage: ../seg input_image initialization_file output_mask" << endl;
		return -1;
	}
	
	Mat in_image;
	in_image = imread(argv[1]/*, CV_LOAD_IMAGE_COLOR*/);

	if (!in_image.data) {
		cout << "Could not load input image!!!" << endl;
		return -1;
	}

	if (in_image.channels() != 3) {
		cout << "Image does not have 3 channels!!! " << in_image.depth() << endl;
		return -1;
	}

	// the output image
	Mat out_image = in_image.clone();

	ifstream f(argv[2]);
	if (!f) {
		cout << "Could not load initial mask file!!!" << endl;
		return -1;
	}

	int width = in_image.cols;
	int height = in_image.rows;

	int n;
	f >> n;

	Mat gaussianOutput;
	Mat greyScale;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;


	//smoothening output
	GaussianBlur(in_image, gaussianOutput, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(gaussianOutput, gaussianOutput, CV_BGR2GRAY);

	Mat input_gray = gaussianOutput.clone();

	normalize(gaussianOutput, gaussianOutput, 0, 255, NORM_MINMAX, CV_8UC1);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Scharr(input_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	Scharr(input_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	//Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs(grad_y, abs_grad_y);

	//average gradient
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, greyScale);

	/* Mat greyScale;
	 cvtColor(in_image, greyScale, CV_BGR2GRAY);

	*/
	VertexPi superSource(-1, -1);
	VertexPi superSink(-2, -2);

	std::vector< std::vector<VertexPi> > aLi(greyScale.rows, vector<VertexPi>(greyScale.cols, VertexPi()));

	for (int i = 0; i < greyScale.rows; i++) {
		for (int j = 0; j < greyScale.cols; j++) {
			aLi.at(i).at(j).ri = i;
			aLi.at(i).at(j).ci = j;
		}
	}

	float maxFlow = 0;
	float sumForSink = 0;
	int numSink = 0;
	float sumForSource = 0;
	float meanSink = 0;
	int numSource = 0;
	float meanSource = 0;
	// get the initil pixels
	for (int i = 0; i < n; ++i) {
		int x, y, t;
		f >> x >> y >> t;

		if (x < 0 || x >= width || y < 0 || y >= height) {
			cout << "I valid pixel mask!" << endl;

			return -1;
		}
		if (t == 0) {
			VertexPi &VertexPi = aLi.at(y).at(x);
			VertexPi.addd(-2, -2, LONG_MAX);
			sumForSink += greyScale.at<float>(y, x);
			numSink++;

		}
		else {
			superSource.addd(y, x, LONG_MAX);
			sumForSource += greyScale.at<float>(y, x);
			numSource++;
		}
	}

	double maxdVal = 0;
	for (int i = 0; i < greyScale.rows; i++) {
		for (int j = 0; j < greyScale.cols; j++) {
			double d_w = 0;
			VertexPi &VertexPi = aLi.at(i).at(j);

			vector<DotI> pixelList;
			if (i > 0) {
				DotI pixelIndex(i - 1, j);
				pixelList.push_back(pixelIndex);
			}
			if (i < height - 1) {
				DotI pixelIndex(i + 1, j);
				pixelList.push_back(pixelIndex);
			}
			if (j > 0) {
				DotI pixelIndex(i, j - 1);
				pixelList.push_back(pixelIndex);
			}
			if (j < width - 1) {
				DotI pixelIndex(i, j + 1);
				pixelList.push_back(pixelIndex);
			}
			for (int pixelIndex = 0; pixelIndex < pixelList.size(); pixelIndex++) {
				double diff = (gaussianOutput.at<uchar>(i, j) - gaussianOutput.at<uchar>(pixelList.at(pixelIndex).a, pixelList.at(pixelIndex).b));
				if (diff < 0.5) {
					d_w = LONG_MAX;
				}
				else {
					d_w = 1;
				}
				VertexPi.addd(pixelList.at(pixelIndex).a, pixelList.at(pixelIndex).b, d_w);
			}
		}
	}
	maxFlow = fordFulkerson(aLi, superSource, superSink, greyScale.rows, greyScale.cols, out_image);

	// write it on disk
	imwrite(argv[3], out_image);

	// also display them both

	namedWindow("Original image", WINDOW_AUTOSIZE);
	namedWindow("Show Marked Pixels", WINDOW_AUTOSIZE);
	imshow("Original image", in_image);
	imshow("Show Marked Pixels", out_image);
	waitKey(0);
	return 0;
};

float fordFulkerson(vector< vector< VertexPi > > &aLi, VertexPi &superSource, VertexPi &superSink, int rows, int cols, Mat &out_image)
{
	float maxFlow = 0;
	int numOfPath = 0;
	while (search(superSource, aLi, superSink)) {
		numOfPath++;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				superSink.isTraversed = false;
				aLi.at(i).at(j).isTraversed = false;
			}
		}
		VertexPi traversalNode = aLi.at(superSink.pri).at(superSink.pci);
		double minFlow = LONG_MAX;
		
		while (!traversalNode.isSource()) {
			VertexPi parentPixel;
			if (traversalNode.pci == -1 && traversalNode.pri == -1)
				parentPixel = superSource;
			else
				parentPixel = aLi.at(traversalNode.pri).at(traversalNode.pci);
			
			minFlow = min(minFlow, parentPixel.getd(traversalNode.ri, traversalNode.ci).weight);
			traversalNode = parentPixel;
		}

		traversalNode = aLi.at(superSink.pri).at(superSink.pci);

		while (true) {
			if (traversalNode.pci == -1 && traversalNode.pri == -1)
				break;

			VertexPi copyParentPixel = aLi.at(traversalNode.pri).at(traversalNode.pci);
			VertexPi &parentPixel = aLi.at(traversalNode.pri).at(traversalNode.pci);
			Distance &fromd = parentPixel.getd(traversalNode.ri, traversalNode.ci);
			Distance &tod = traversalNode.getd(parentPixel.ri, parentPixel.ci);
			fromd.weight -= minFlow;
			tod.weight += minFlow;

			traversalNode = copyParentPixel;
		}
		maxFlow += minFlow;
		//        cout << endl << "MINFLOW::" << minFlow << endl;
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Vec3b pixel = out_image.at<Vec3b>(i, j);
			if (aLi.at(i).at(j).isTraversed) {
				pixel[0] = 255;
				pixel[1] = 255;
				pixel[2] = 255;
			}
			else {
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 0;
			}
			out_image.at<Vec3b>(i, j) = pixel;
		}
	}
	//    cout<<endl<<"NumOfPath"<<numOfPath<<endl;
	return maxFlow;
}

bool search(VertexPi dot, vector< vector< VertexPi > > &aLi, VertexPi &siNode) {
	queue < VertexPi > q;
	queue < VertexPi > empty;
	q.push(dot);
	while (!q.empty())
	{
		VertexPi u = q.front();
		q.pop();
		for (int v = 0; v < u.dList.size(); v++)
		{
			Distance d = u.dList.at(v);
			if (d.tri >= 0 && d.tci >= 0) {
				VertexPi &VertexPi = aLi.at(d.tri).at(d.tci);
				if (!VertexPi.isTraversed && d.weight > 0)
				{
					VertexPi.isTraversed = true;
					VertexPi.setParent(u.ri, u.ci);
					q.push(VertexPi);
				}
			}
			else if (d.isToSink() && d.weight > 0) {
				siNode.isTraversed = true;
				siNode.setParent(u.ri, u.ci);
				swap(q, empty);
				return true;
			}
		}
	}
	return siNode.isTraversed;
}

