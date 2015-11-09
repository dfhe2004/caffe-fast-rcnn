#include "caffe/caffe.hpp"
#include <string>
#include <vector>
using namespace caffe;

void main(){

	char* proto = "H:\\workspace\\cnn\\mnist\\lenet_auto_test.prototxt";
	char* model = "H:\\workspace\\cnn\\mnist\\lenet_iter_5000.caffemodel";

	Caffe::set_mode(Caffe::CPU);

	boost::shared_ptr< Net<float> > net(new caffe::Net<float>(proto, TEST));

	net->CopyTrainedLayersFrom(model);
	std::cout<<"--- over ---"<<std::endl;

}
