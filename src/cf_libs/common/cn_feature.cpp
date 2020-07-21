#include "cn_feature.hpp"



void ColorName::load_w2c()
{
	// load the normalized Color Name matrix
    std::string w2crs_path("/Users/leon/Gallopwave/cf_tracking/src/cf_libs/common/w2crs.txt");
	std::ifstream ifstr(w2crs_path);
    if(!ifstr){
        std::cerr<<"cant not find "<<w2crs_path<<std::endl;
        exit(0);
    }

	for (int i = 0; i < 10; i++)
	{
		w2c_t.push_back(std::vector<double>(32768,0));
	}
	std::vector<double> tmp(10, 0);
	for (int i = 0; i < 32768; i++)
	{
		w2c.push_back(tmp);
	}
	double tmp_val;
	for (int i = 0; i < 32768; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			ifstr >> tmp_val;
			w2c[i][j] = w2c_t[j][i] = tmp_val;
		}
	}
	ifstr.close();
}

template<typename T>
std::vector<T> ColorName::get_max(std::vector<std::vector<T> > &inp, int dim)
{
	// dim = 1 max row, 2 max column
	std::vector<T> ret;
	if (dim == 1)
	{
		ret.reserve( inp[0].size());
		int inp0_size = inp[0].size();
		for (int j = 0; j < inp0_size; j++)
		{
			ret.push_back(inp[0][j]);
			int inp_size = inp.size();
			for (int i = 1; i < inp_size; i++)
			{
				if (inp[i][j] > ret[j])
				{
					ret[j] = inp[i][j];
				}
			}
		}
	}
	else if (dim == 2)
	{
		ret.reserve( inp.size());
		int inp_size = inp.size();
		for (int i = 0; i < inp_size; i++)
		{
			ret.push_back(*max_element(inp[i].begin(), inp[i].end()));
		}
	}
	return ret;
}
std::vector<double> ColorName::get_vector(int dim, int ind)
{
	// dim = 1  row, 2  column
	if (dim == 2)
	{
		return w2c_t[ind];
	}
	else// if (dim == 1)
	{
		return w2c[ind];
	}
}
cv::Mat ColorName::reshape(std::vector<double> &inp, int rows, int cols)
{

	cv::Mat result(rows, cols, CV_64FC1);
	double* data = ((double*)result.data);
	memcpy(data, ((double*)(&inp[0])), rows*cols*sizeof(double));
	return result;
}
std::vector<double> ColorName::select_indeces(std::vector<double> &inp, std::vector<int> &indeces)
{
	int sz = std::min(inp.size(), indeces.size());
	std::vector<double> res(sz, 0);
	for (int i = 0; i < sz; i++)
	{
		res[i] = inp[indeces[i]];
	}
	return res;
}

std::vector<cv::Mat> ColorName::im2c(cv::Mat &im, std::vector<std::vector<double>> &w2c, int color)
{
	std::vector<cv::Mat> out;
	// input im should be DOUBLE !
	// color=0 is color names out
	// color=-1 is colored image with color names out
	// color=1-11 is prob of colorname=color out;
	// color=-1 return probabilities
	// order of color names: 
	double color_values[][3] = { 
		{  0,  0,  0 },   // black
		{  0,  0,  1 },   // blue
		{ .5, .4, .25 },  // brown
		{ .5, .5, .5 },   // grey
		{  0,  1,  0 },   // green
		{  1, .8,  0 },   // orange
		{  1, .5,  1 },   // pink
		{  1,  0,  1 },   // pruple
		{  1,  0,  0 },   // red
		{  1,  1,  1 },   // white
		{  1,  1,  0 }    // yellow     
	};

	std::vector<cv::Mat> im_split;
	cv::split(im, im_split);
	cv::Mat RR = im_split[2];
	cv::Mat GG = im_split[1];
	cv::Mat BB = im_split[0];
	
	double*  RRdata = ((double*)RR.data), *GGdata = ((double*)GG.data), *BBdata = ((double*)BB.data);
	int w = RR.cols;
	int h = RR.rows;
	std::vector<int> index_im(w * h, 0);
	int l = index_im.size();

	for (int i = 0; i < l; i++)
	{
		//int j = (i / w) + (i//w) * h;
		// I don't need +1 in the formula because the indeces are zero based here
		index_im[i] = (int)(floor(RRdata[i] / 8) + 32 * floor(GGdata[i] / 8) + 32 * 32 * floor(BBdata[i] / 8));
	}

	if (color == 0)
	{
		std::vector<double> w2cM = get_max(w2c, 2);
		std::vector<double> selected = select_indeces(w2cM, index_im);
		out.push_back(reshape(selected, im.rows, im.cols));
	}

	if (color > 0 && color < 12)
	{
		std::vector<double> w2cM = get_vector(2, color - 1);
		std::vector<double> selected = select_indeces(w2cM, index_im);
		out.push_back(reshape(selected, im.rows, im.cols));
	}

	if (color == -1)
	{
		out.push_back(im);
		std::vector<double> w2cM = get_max(w2c, 2);
		std::vector<double> temp = select_indeces(w2cM, index_im);
		cv::Mat out2 = reshape(temp, im.rows, im.cols);
	}

	if (color == -2)
	{
		for (int i = 0; i < 10; i++)
		{
			std::vector<double> vec = get_vector(2, i);
			std::vector<double> selected = select_indeces(vec, index_im);
			cv::Mat temp = reshape(selected, im.rows, im.cols);
			out.push_back(temp);
		}
	}

	return out;
}

ColorName::ColorName(ColorNameParameters& param): 
	_param(param), is_initialized(false), is_update(false)
{
    if (w2c.size() == 0)
	{
		load_w2c();
	}
    
    if(param.num_compressed_dim > 10){
        std::cerr<<"param.num_compressed_dim should smaller than 10"<<std::endl;
        exit(0);
    }
}


cv::Mat ColorName::get_feature_map(cv::Mat &im_patch)
{
	// the dimension of the valid features
    int num_feature_levels = 10;
	// allocate space (for speed)
	std::vector<cv::Mat> out(num_feature_levels, cv::Mat::zeros(im_patch.rows, im_patch.cols, CV_64FC1));
	
    // Color Names are available for three-channel images only (color sequances)
    cv::Mat double_patch;
	im_patch.convertTo(double_patch, CV_64FC1);
	out = im2c(double_patch, w2c, -2);

	if(_param.visualization){
		vis_out = out;
	}
	
    // reshape form vector<Mat>: "out" to Mat(dim x pixel_num): "out_pca"
    int total_len = target_sz.width*target_sz.height;
	cv::Mat out_pca = cv::Mat::zeros(out.size(), total_len, CV_64FC1);
		int ind = 0;
		double* data = ((double*)out_pca.data);
		int out_size = out.size();
		for (int i = 0; i < out_size; i++)
		{
			cv::Mat tmp = out[i].t();
			memcpy(data + i * total_len, tmp.data, total_len * sizeof(double));
		}
		out_pca = out_pca.t();
        /* out_pca: pixel_num x dim */ 
	return out_pca;
}


std::vector<cv::Mat> ColorName::feature_projection(cv::Mat &x_pca, cv::Mat& projection_matrix)
{
	// Calculates the compressed feature map by mapping the PCA features with
	// the projection matrix.
	std::vector<cv::Mat> z;
    z.reserve(_param.num_compressed_dim);

    // project the PCA-features using the projection matrix and reshape to a window
    /* x_pca: [ pixel x dim ] */ 
	cv::Mat tmp = (x_pca * projection_matrix);
	for (int i = 0; i < tmp.cols; i++)
	{
		cv::Mat tmpCol = tmp.col(i).clone();
		cv::Mat tmpCol2(target_sz.width, target_sz.height, CV_64FC1);
		memcpy(tmpCol2.data, tmpCol.data, tmp.rows * sizeof(double));
		// concatinate the feature windows
		z.push_back(tmpCol2.t());
	} 

	return z;
}


void ColorName::dimensionality_reduction(){
	if(is_initialized == false)
	{
		data_matrix = cv::Mat::zeros(z_pca.rows, z_pca.cols, CV_64FC1);
	}

    // compute the mean appearance
	/* claculate mean of each col */
    /* z_pca: */
	cv::reduce(z_pca, data_mean, 0, cv::REDUCE_AVG);
    
    // substract the mean from the appearance to get the data matrix
	double*data = ((double*)data_matrix.data);
	for (int i = 0; i < z_pca.rows; i++)
	{
		memcpy(data + i * z_pca.cols, ((cv::Mat)(z_pca.row(i) - data_mean)).data, z_pca.cols * sizeof(double));
	}

	// calculate the covariance matrix
	cov_matrix = (1.0 / (target_sz.width * target_sz.height - 1))* (data_matrix.t() * data_matrix);

    cv::Mat vt; /* pca_basis^T */
    cv::Mat R_p; 

    if(is_initialized == false){
        R_p = cov_matrix;
    }
    else{
        R_p = (1 - _param.compression_learning_rate) * old_cov_matrix + _param.compression_learning_rate * cov_matrix;
    }
    cv::SVD::compute(R_p, pca_variances, pca_basis, vt);

    // calculate the projection matrix as the first principal
	// components and extract their corresponding variances
	projection_matrix = pca_basis(cv::Rect(0, 0, _param.num_compressed_dim, pca_basis.rows)).clone();

	/* Λ_p in Algorithm.(1) */
	cv::Mat projection_variances = cv::Mat::zeros(_param.num_compressed_dim, _param.num_compressed_dim, CV_64FC1);
	for (int i = 0; i < _param.num_compressed_dim; i++)
	{
		((double*)projection_variances.data)[i + i*_param.num_compressed_dim] = ((double*)pca_variances.data)[i];
	}

    if(is_initialized == false){
		old_cov_matrix = projection_matrix * projection_variances * projection_matrix.t();
    }
    else{
		old_cov_matrix =
			    (1 - _param.compression_learning_rate) * old_cov_matrix +
				_param.compression_learning_rate * (projection_matrix * projection_variances * projection_matrix.t());
    }
}


std::vector<cv::Mat> ColorName::init(cv::Mat &im_patch){
    target_sz = im_patch.size();
	patch = im_patch.clone();
    cur_x_pca = get_feature_map(patch);
    
	z_pca = cur_x_pca;
    dimensionality_reduction();

	is_initialized = true;
    return feature_projection(cur_x_pca, projection_matrix);;
}

std::vector<cv::Mat> ColorName::update(cv::Mat &im_patch){
	if( target_sz != im_patch.size() )
	{
		cv::resize(im_patch, patch, target_sz, 0, 0, _param.resizeType);
	}
	else
	{
		patch = im_patch.clone();
	}
    cur_x_pca = get_feature_map(patch);
	
	if( is_update == false)
	{
		return feature_projection(cur_x_pca, projection_matrix);		
	}
	// if is_update == true
    z_pca = (1 - _param.learning_rate) * z_pca + _param.learning_rate * cur_x_pca;
    dimensionality_reduction();

	std::vector<cv::Mat> result = feature_projection(z_pca, projection_matrix);
	if(_param.visualization && is_update)
	{
		for(size_t i=0; i<result.size(); i++)
		{
			std::string win_name = "pca_"+std::to_string(i);
			cv::imshow(win_name, result[i]);
		}
		cv::imshow("patch", patch);
	}
	is_update = false;

    // return feature_projection(z_pca, projection_matrix);
	return result;

}



