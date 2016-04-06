#include <cmath>
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        top[0]->ReshapeLike(*bottom[0]);
        CHECK_EQ(bottom[1]->shape(1), 6) << "Second blob should be 6-dimension theta"; //   (batch, 2,3)
        num_ = bottom[0]->shape()[0];
        channel_ = bottom[0]->shape()[1];
        height_ = bottom[0]->shape()[2];
        width_ = bottom[0]->shape()[3];
	    CHECK_GE(height_*width_,4)<<"(height,width) >= (2,2)";

        map_size_ = width_ * height_;

        // init target coordinate
        vector<int> target_shape;
        target_shape.push_back(1);
        target_shape.push_back(3);
        target_shape.push_back(height_);
        target_shape.push_back(width_);
        target_.Reshape(target_shape);
        Dtype* target_data = target_.mutable_cpu_data();

        for (int h = 0; h < height_; ++h) {   // target_data:  [-1,1]
            for (int w = 0; w < width_; ++w) {
                // for x;
                target_data[target_.offset(0, 0, h, w)] = (Dtype) w / (Dtype) (width_ - 1) * 2. - 1.;
                // for y
                target_data[target_.offset(0, 1, h, w)] = (Dtype) h / (Dtype) (height_ - 1) * 2. - 1.;
                // for constant
                target_data[target_.offset(0, 2, h, w)] = (Dtype) 1.0;
            }
        }

        // create source coordinates
        vector<int> source_shape;
        source_shape.push_back(num_);
        source_shape.push_back(2);
        source_shape.push_back(height_);
        source_shape.push_back(width_);
        source_.Reshape(source_shape);
        
        // create source range for bilinear sampling
        vector<int> source_range_shape;
        source_range_shape.push_back(num_);
        source_range_shape.push_back(height_);
        source_range_shape.push_back(width_);
        source_range_shape.push_back(2);
        source_range_.Reshape(source_range_shape);
        //caffe_set<Dtype>(num_*map_size_, -1, source_range_.mutable_cpu_data());
        
        // create source gradient cache for different channels, use for gpu calculation
        vector<int> source_grad_shape;
        source_grad_shape.push_back(channel_);
        source_grad_shape.push_back(num_);
        source_grad_shape.push_back(2);
        source_grad_shape.push_back(height_);
        source_grad_shape.push_back(width_);
        source_grad_cache_.Reshape(source_grad_shape);
        
        vector<int> all_ones_shape;
        all_ones_shape.push_back(channel_);
        source_grad_op_.Reshape(all_ones_shape);
        caffe_set<Dtype>(channel_, 1, source_grad_op_.mutable_cpu_data());
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        Dtype* top_data = top[0]->mutable_cpu_data();
        const Dtype* theta_data = bottom[1]->cpu_data();
        const Dtype* target_data = target_.cpu_data();

        Dtype* source_data = source_.mutable_cpu_data();
        int* source_range_data = source_range_.mutable_cpu_data();
        
		caffe_set<Dtype>(top[0]->count(), 0, top_data);
        caffe_set<int>(num_*map_size_, -1, source_range_.mutable_cpu_data());

		for (int n = 0; n < num_; ++n) {
            // compute source coordinate 
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, map_size_, 3, Dtype(1.0),
                    theta_data + n * 6, target_data, Dtype(0.), source_data + n * 2 * map_size_);
            // compute source in real source coordinate range, [0,w)x[0,h)
            caffe_add_scalar(2 * map_size_, (Dtype)1. , source_data + n * 2 * map_size_);
            caffe_scal<Dtype>(map_size_, (Dtype) (width_ - 1) / (Dtype) 2., source_data + n * 2 * map_size_);
            caffe_scal<Dtype>(map_size_, (Dtype) (height_ - 1) / (Dtype) 2., source_data + n*2*map_size_+map_size_);
            
            
            // compute U given source coordinate: O(W*H)
            for (int h = 0; h < height_; ++h) {
                for (int w = 0; w < width_; ++w) {
                    Dtype x = source_data[source_.offset(n, 0, h, w)];
                    Dtype y = source_data[source_.offset(n, 1, h, w)];

                    //O(C)
                    int w_min = std::max<Dtype>(0, floor(x)); 
                    int w_max = std::min<Dtype>(width_-1, ceil(x)); 
                    int h_min = std::max<Dtype>(0,floor(y)); 
                    int h_max = std::min<Dtype>(height_-1, ceil(y)); 
					
					if(w_max<w_min || h_max<h_min)  continue;
					source_range_data[source_range_.offset(n,h,w,0)] = w_min;
                    source_range_data[source_range_.offset(n,h,w,1)] = h_min;

					Dtype alpha = x-w_min;
					Dtype beta  = y-h_min;
					for(int c=0; c<channel_; ++c){
						Dtype T00 = bottom[0]->data_at(n, c, h_min, w_min);
						Dtype T10 = bottom[0]->data_at(n, c, h_min+1, w_min);
						Dtype T11 = bottom[0]->data_at(n, c, h_min+1, w_min+1);
						Dtype T01 = bottom[0]->data_at(n, c, h_min, w_min+1);
						top_data[top[0]->offset(n, c, h, w)] = T00 + alpha*(T10 - T00) + beta*(T01 - T00) + alpha*beta*(T00+T11-T10-T01);
					}
				}
            }
        }
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
        // @IMPRV current version ignores propagate_down signal
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* data_diff = bottom[0]->mutable_cpu_diff();
        Dtype* theta_diff = bottom[1]->mutable_cpu_diff();
                const Dtype* target_data = target_.cpu_data();
        const Dtype* source_data = source_.cpu_data();
        const int* source_range_data = source_range_.cpu_data();
        Dtype* source_diff = source_.mutable_cpu_diff();

        caffe_set<Dtype>(bottom[0]->count(), 0, data_diff);
//        caffe_set<Dtype>(source_.count(), 0, source_diff);

        const Dtype width_const = (Dtype)(width_ - 1) / (Dtype)2.;
        const Dtype height_const = (Dtype)(height_ - 1) / (Dtype)2.;
        for (int n = 0; n < num_; ++n) {
            for (int h = 0; h < height_; ++h) {
                for (int w = 0; w < width_; ++w) {
                    Dtype x = source_data[source_.offset(n, 0, h, w)];
                    Dtype y = source_data[source_.offset(n, 1, h, w)];
                    int w_min = source_range_data[source_range_.offset(n,h,w,0)];
					if (w_min<0) continue;
                    int h_min = source_range_data[source_range_.offset(n,h,w,1)];
                    
					Dtype tmp_source_x = 0;
                    Dtype tmp_source_y = 0;
                  
                    for (int c = 0; c < channel_; ++c) {
						for (int hh = h_min; hh <= h_min+1; ++hh) {
							for (int ww = w_min; ww <= w_min+1; ++ww) {
								int sign_x = caffe_sign<Dtype>(ww - x);
								int sign_y = caffe_sign<Dtype>(hh - y);//(y <= (Dtype)hh ) ? 1 : -1;

                                // d(L)/d(U^{c}_{nm})=\sum_{j} d(L)/d(V^{c}_{j}) * d(V^{c}_{j})/d(U^{c}_{nm})
                                // bottom_diff[(n,c,hh,ww)]=\sum_{j} top_diff[(n,c,h,w)] * eq(6) (an error)
                                Dtype buffer = top_diff[top[0]->offset(n, c, h, w)];
                                data_diff[bottom[0]->offset(n, c, hh, ww)] += buffer * (1 - fabs(x - ww)) * (1 - fabs(y - hh));
                                // d(L)/d(x_{j})=\sum_{c} d(L)/d(V^{c}_j)*d(V^{c}_j)/d(x_{j})
                                // source_diff[(n,0,h,w)] = \sum_{c} top[(n,c,h,w)] * \sum_{nm} U_{nm} max
                                buffer *= bottom[0]->data_at(n,c,hh,ww);
                                tmp_source_x += buffer*(1-fabs(y-hh))*sign_x;
                                tmp_source_y += buffer*(1-fabs(x-ww))*sign_y;
                            }
                        }
                    }
                    source_diff[source_.offset(n,0,h,w)] = tmp_source_x* width_const; 
                    source_diff[source_.offset(n,1,h,w)] = tmp_source_y* height_const;
                }
            }
            // d(L)/d(theta)
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2, 3, map_size_,
                    (Dtype)1., source_diff + n * 2 * map_size_, target_data, (Dtype)0., theta_diff + n * 6);
        }
    }


#ifdef CPU_ONLY
    STUB_GPU(SpatialTransformerLayer);
#endif

    INSTANTIATE_CLASS(SpatialTransformerLayer);
	REGISTER_LAYER_CLASS(SpatialTransformer);
} // namespace caffe
