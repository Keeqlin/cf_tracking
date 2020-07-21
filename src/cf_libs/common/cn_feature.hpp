
#ifndef _COLOR_NAME_FEATURE_HPP_
#define _COLOR_NAME_FEATURE_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv_modules.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>

// #include "math_helper.hpp"

class ColorNameParameters{
public:
   /* parameters setting according to the paper 
	* Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg and Joost van de Weijer. 
    * "Adaptive Color Attributes for Real-Time Visual Tracking". 
	* Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.
	*/
    double learning_rate  = 0.075;
    double compression_learning_rate = 0.15;  // learning rate for the adaptive dimensionality reduction(denoted "mu" in the paper)
	int num_compressed_dim = 3;               // the dimensionality of the compressed features
	int visualization = 0;                    // params.visualization = 1;
    int resizeType = cv::INTER_LINEAR;
};


class ColorName{
private:
    ColorNameParameters _param;
    bool is_initialized;
    bool is_update;

    cv::Mat patch;
    cv::Size target_sz;
    std::vector<std::vector<double>> w2c;
	std::vector<std::vector<double>> w2c_t; // transpose

    cv::Mat cur_x_pca; 
    cv::Mat z_pca;
    cv::Mat pca_variances;   /* S_p in Algorithm.(1) */
    cv::Mat pca_basis;       /* E_p in Algorithm.(1) */
	cv::Mat old_cov_matrix;  /* Q_p in Algorithm.(1) */
    
    // initialize the projection matrix
    cv::Mat projection_matrix;  /*B_p in eq.(7) */

    // for dimensionality_reduction
    cv::Mat data_mean, cov_matrix, data_matrix;


    // visualization
    std::vector<cv::Mat> vis_out;

    /* extracting Color Name (dim:10) */
    void load_w2c();
    template<typename T>
	std::vector<T> get_max(std::vector<std::vector<T> > &inp, int dim);
	std::vector<double> get_vector(int dim, int ind);
	cv::Mat reshape(std::vector<double> &inp, int rows, int cols);
	std::vector<double> select_indeces(std::vector<double> &inp, std::vector<int> &indeces);
    std::vector<cv::Mat> im2c(cv::Mat &im, std::vector<std::vector<double>> &w2c, int color = -2);
    cv::Mat get_feature_map(cv::Mat &im_patch);

    /* PCA-part */
    std::vector<cv::Mat> feature_projection(cv::Mat& x_pca, cv::Mat& projection_matrix);
    void dimensionality_reduction();  /* update projection_matrix*/

public:
    ColorName(ColorNameParameters &param);
    std::vector<cv::Mat> init(cv::Mat &im_patch);
    std::vector<cv::Mat> update(cv::Mat &im_patch);
    void set_update(){
        is_update = true;
    }
    int get_compressed_dim(){
        return _param.num_compressed_dim;
    }
};

template<typename T>
void cnToCvCol(std::vector<cv::Mat>& cn_feature, cv::Mat& cvFeatures, int colIdx, T cosFactor)
{
    // if( cvFeatures.cols-1 < colIdx)
    // {
    //     std::cerr<<"cvFeatures.cols-1("<<std::to_string(cvFeatures.cols-1)<<") < colIdx("<<std::to_string(colIdx)<<")"<<std::endl;
    //     exit(0);
    // }

    for(auto& cnf: cn_feature)
    {
        cnf = cnf.reshape(1, cnf.total());
    }
    cv::Mat Feature_col;
    cv::vconcat(cn_feature, Feature_col);

    Feature_col *= cosFactor;
    Feature_col.col(0).copyTo(cvFeatures.col(colIdx));
}

#endif // _COLOR_NAME_FEATURE_HPP_