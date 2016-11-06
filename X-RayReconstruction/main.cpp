#include <opencv2\opencv.hpp>
#include "CVUtil.h"
//#pragma region OPENCV3_LIBRARY_LINKER
//#ifdef _DEBUG
//#define CV_EXT "d.lib"
//#else
//#define CV_EXT ".lib"
//#endif
//#define CV_VER  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
//#pragma comment(lib, "opencv_world" CV_VER CV_EXT)
//#pragma endregion

using namespace cv;
using namespace std;


int main(void)
{
	//	1. �摜�̓ǂݍ���
	Mat src_read = imread("doge.jpg", IMREAD_GRAYSCALE);
	Mat src, srcf, srcd;		//	���ۂ̉��Z��srcf�̉�f�l���g��
	resize(src_read, src, Size(640, 480), INTER_CUBIC);
	src.convertTo(srcf, CV_32F); src.convertTo(srcd, CV_64F);
	cout << "loaded image information:\n"
		<< "image size = " << src.size()
		<< endl;
	imshow("test", src);
	waitKey();
	//	�摜�̕ۑ�
	imwrite("input.png", src);

	//---------------------------------------
	//	2. 1����X�����e����S�������ŎB�e
	//---------------------------------------
	const int div_rotation = 360;						//	360�������ĎB�e
	Mat projectedImage = Mat(div_rotation, 880, CV_64FC1, Scalar::all(0));	//	���e���x�N�g���̏W�� �s�����p�x
	cout << "���e���T�C�Y : " << projectedImage.size() << endl;
	//	��]�ptheta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	�p�xrad�ɕϊ��i�S����180deg�j
		//	�摜����]
		Mat rotatedImage;
		int offsety = projectedImage.cols - srcd.rows;
		int offsetx = projectedImage.cols - srcd.cols;
		copyMakeBorder(srcd, rotatedImage, offsety / 2, offsety / 2, offsetx / 2, offsetx / 2, BORDER_CONSTANT, Scalar::all(0));//	�摜���S�Ɖ�]���S�����킹��
		Mat affine = getRotationMatrix2D(Point2d(rotatedImage.cols / 2, rotatedImage.rows / 2), -theta / CV_PI * 180, 1.0);		//	�A�t�B���ϊ��s��
		warpAffine(rotatedImage, rotatedImage, affine, rotatedImage.size(), 1, 0, Scalar::all(0));
		//	�摜������X���𓖂Ăĉ摜�㕔�ŎB�e
		Mat reduceRow(Size(rotatedImage.cols,1), CV_64FC1, Scalar::all(0));
		reduce(rotatedImage, reduceRow, 0, REDUCE_SUM);	//�@�s�����̂܂܉��Z
		reduceRow.copyTo(projectedImage.row(th));
		//	�B�e���摜��\��
		normalize(rotatedImage, rotatedImage, 0, 1, CV_MINMAX);
		imshow("test", rotatedImage);
		Mat projectedImage_scaled;
		normalize(projectedImage, projectedImage_scaled, 0, 255, CV_MINMAX);
		projectedImage_scaled.convertTo(projectedImage_scaled, CV_8UC1);
		imshow("X�����e��", projectedImage_scaled);
		cout << "Captureing... :" << th << " / " << div_rotation << "\r";
		waitKey(1);
	}
	imshow("test", src);
	cout << endl << "Capture finished!!" << endl;
	waitKey();
	//	�摜�̕ۑ�
	Mat projectedImage_scaled;
	normalize(projectedImage, projectedImage_scaled, 0, 255, CV_MINMAX);
	projectedImage_scaled.convertTo(projectedImage_scaled, CV_8U);
	imwrite("x-ray_projection.png", projectedImage_scaled);

	//---------------------------------------------------------------------------------------------
	//	3. ���e������̍č\���i�t�[���G�ϊ��@�j
	//	���e���̊p�x�Œ�1����(r)�t�[���G�ϊ������摜��2����(x,y)�t�[���G�ϊ��@�Ƃ������w�I�����i���e�藝�j�𗘗p����D
	//	�p�xth�ł̓��e���̂P�����t�[���G�ϊ����ʂ��C���g������uv��̊p�xth�̕����ɕ��ׂĂ���
	//	����uv���Q�����t�t�[���G�ϊ�����ƌ��摜�����������D
	//---------------------------------------------------------------------------------------------
	//	3.0 �܂���DFT�Ɋ���邽�߂Ɍ��摜��FFT���s���i�����̊w�K�p�j�D
	//	DFT�ɍœK�ȃT�C�Y���擾�i���摜���傫���j
	Size optDFTSize(projectedImage.cols, projectedImage.cols);//(getOptimalDFTSize(projectedImage.cols),getOptimalDFTSize(projectedImage.cols));
	Mat optDFTImg;		//	���摜�̗]���0�Ŗ��߂��摜
	int offsety = projectedImage.cols - srcd.rows;
	int offsetx = projectedImage.cols - srcd.cols;
	copyMakeBorder(srcd, optDFTImg, offsety / 2, offsety / 2, offsetx / 2, offsetx / 2, BORDER_CONSTANT, Scalar::all(0));
	//���f���摜complexImg�i����ch/����ch��2ch�j�𐶐�
	Mat complexPlanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTImg.size(), CV_64F) };
	Mat complexImg;		//	������DFT�̌��ʂ�����
	merge(complexPlanes, 2, complexImg);
	//	DFT���s�i���ۂɂ�FFT�j
	dft(complexImg, complexImg);
	cvutil::fftShift(complexImg, complexImg);		//	FFT���ʂ͏ی������]���Ă���̂Œ���
	//	DFT���ʕ\���p�Ƀp���[�X�y�N�g���摜�ɕϊ�
	//	���f����L2�m�����̑ΐ�
	split(complexImg, complexPlanes);
	Mat magImg;
	magnitude(complexPlanes[0], complexPlanes[1], magImg);		//	���f���v�f�̃m�������i�[
	magImg += Scalar::all(1);	//	�ΐ����̂��߂̃I�t�Z�b�g
	log(magImg, magImg);		//	�l��ΐ���
	normalize(magImg, magImg, 0, 1, CV_MINMAX);
	imshow("���摜��FFT����", magImg);
	//	���ʂ̕ۑ�
	normalize(magImg, magImg, 0, 255, CV_MINMAX);
	magImg.convertTo(magImg, CV_8UC1);
	imwrite("FFT_spectrum_from_src.png", magImg);
	//	�t�t�[���G�ϊ�
	Mat invDFTImg = complexImg;
	Mat invDFTplanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTImg.size(), CV_64F) };
	cvutil::fftShift(invDFTImg, complexImg);
	dft(complexImg, invDFTImg, DFT_INVERSE + DFT_SCALE);	//	FFT���ʂ����̂܂܋tFFT�ɂ�����
	split(invDFTImg, invDFTplanes);		//	�tDFT�̌��ʂ͎����ɏo�Ă���
	Mat invDFTImg_scaled;
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 1, CV_MINMAX);
	imshow("���摜�̋tFFT����", invDFTImg_scaled);
	//	���ʂ̕ۑ�
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 255, CV_MINMAX);
	invDFTImg_scaled.convertTo(invDFTImg_scaled, CV_8UC1);
	imwrite("inverseFFT_from_src.png", invDFTImg_scaled);
	waitKey();

	//	1. �B�e�摜g(r, th)��1�����t�[���G�ϊ�����G(s, th)�𓾂�
	double optDFTSize_rth = projectedImage.cols;//getOptimalDFTSize(projectedImage.cols);
	Mat optDFTImage_rth;
	copyMakeBorder(projectedImage, optDFTImage_rth, 0, 0, 0, optDFTSize_rth - projectedImage.cols, BORDER_CONSTANT, Scalar::all(0));
	//	���f���摜���쐬
	Mat complexDFTPlanes_rth[] = { Mat_<double>(optDFTImage_rth), Mat::zeros(optDFTImage_rth.size(), CV_64F) };
	Mat complexDFTImage_rth;
	merge(complexDFTPlanes_rth, 2, complexDFTImage_rth);
	//	1����DFT���s
	dft(complexDFTImage_rth, complexDFTImage_rth, DFT_ROWS);
	cvutil::fftShift1D(complexDFTImage_rth, complexDFTImage_rth);	//	FFT�摜���]
	split(complexDFTImage_rth, complexDFTPlanes_rth);
	//	DFT���ʕ\��
	Mat magImage_rth;
	magnitude(complexDFTPlanes_rth[0], complexDFTPlanes_rth[1], magImage_rth);
	magImage_rth += Scalar::all(1);
	log(magImage_rth, magImage_rth);
	normalize(magImage_rth, magImage_rth, 0, 1, CV_MINMAX);
	imshow("X�����e��1����FFT����", magImage_rth);
	//	���ʂ̕ۑ�
	normalize(magImage_rth, magImage_rth, 0, 255, CV_MINMAX);
	magImage_rth.convertTo(magImage_rth, CV_8UC1);
	imwrite("FFT1D_projection_rth.png", magImage_rth);

	//	2. G(s, th)��F(u,v)��Ԃɕ��בւ���
	//	u = s * cos(th), v = s * sin(th)
	Size optDFTSize_uv(optDFTSize_rth, optDFTSize_rth);
	Mat complexDFTPlanes_uv[] 
		= { Mat::zeros(optDFTSize_uv, CV_64FC1), Mat::zeros(optDFTSize_uv, CV_64FC1) };
	//	�T�u�s�N�Z���T���v�����O�̂��߂�float�^�ɕϊ�
	Mat complexDFTPlane_sth_re, complexDFTPlane_sth_im;
	complexDFTPlanes_rth[0].convertTo(complexDFTPlane_sth_re, CV_32F);
	complexDFTPlanes_rth[1].convertTo(complexDFTPlane_sth_im, CV_32F);
	for (int i = 0; i < optDFTSize_uv.height; i++) {
		for (int j = 0; j < optDFTSize_uv.width; j++) {
			Point2d center_uv(optDFTSize_uv.width / 2, optDFTSize_uv.height / 2);
			Point2d p_uv(j - center_uv.x, - i + center_uv.y);	//	���݂̃X�R�[�v���W(u,v)
			double theta = atan2(p_uv.y, p_uv.x);		//	[-PI, PI]
			//double radius = p_uv.x * cos(theta) + p_uv.y * sin(theta);
			double radius = norm(p_uv);
			Point2d p_sth;					//	p_uv�ɑΉ�����G(s, th)�ł̃T���v���_
			//	���_����̋��������e����r�������傫���ꍇ��0�Ƃ���
			if (radius > optDFTSize_rth / 2.0) {
				complexDFTPlanes_uv[0].at<double>(i, j) = 0.0;	//	����
				complexDFTPlanes_uv[1].at<double>(i, j) = 0.0;	//	����
				continue;
			}
			//	p_uv��ϐ��ϊ�����p_sth�ɑ��
			p_sth.x = (theta > 0) ? center_uv.x + radius : center_uv.x - radius;
			p_sth.y = (theta > 0) ? theta*div_rotation / CV_PI : (theta + CV_PI) * div_rotation / CV_PI;
			complexDFTPlanes_uv[0].at<double>(i, j) = cvutil::sampleSubPix(complexDFTPlane_sth_re, p_sth);
			complexDFTPlanes_uv[1].at<double>(i, j) = cvutil::sampleSubPix(complexDFTPlane_sth_im, p_sth);
		}
	}
	Mat complexDFTImage_uv;
	merge(complexDFTPlanes_uv, 2, complexDFTImage_uv);
	//	DFT���ʕ\��
	split(complexDFTImage_uv, complexDFTPlanes_uv);
	Mat magImage_uv;
	magnitude(complexDFTPlanes_uv[0], complexDFTPlanes_uv[1], magImage_uv);
	magImage_uv += Scalar::all(1);
	log(magImage_uv, magImage_uv);
	normalize(magImage_uv, magImage_uv, 0, 1, CV_MINMAX);
	imshow("X�����e��2����FFT��������", magImage_uv);
	//	���ʂ̕ۑ�
	normalize(magImage_uv, magImage_uv, 0, 255, CV_MINMAX);
	magImage_uv.convertTo(magImage_uv, CV_8UC1);
	imwrite("FFT1D_projection_xy.png", magImage_uv);

	//	3. F(u,v)����f(x,y)�𕜌�
	Mat complexInvDFTImage;
	cvutil::fftShift(complexDFTImage_uv, complexDFTImage_uv);
	dft(complexDFTImage_uv, complexInvDFTImage, DFT_INVERSE + DFT_SCALE);
	Mat complexInvDFTPlanes[] = { Mat::zeros(optDFTSize_uv, CV_64F), Mat::zeros(optDFTSize_uv, CV_64F) };
	split(complexInvDFTImage, complexInvDFTPlanes);
	Mat invImage_scaled;
	normalize(complexInvDFTPlanes[0], invImage_scaled, 0, 1, CV_MINMAX);
	imshow("���e������̕�������", invImage_scaled);
	//	���ʂ̕ۑ�
	normalize(invImage_scaled, invImage_scaled, 0, 255, CV_MINMAX);
	invImage_scaled.convertTo(invImage_scaled, CV_8UC1);
	imwrite("inverseFFT2D_projection_xy.png", invImage_scaled);
	waitKey();

	//---------------------------------------------------------------------------------------------
	//	4. ���e������̍č\���i�t�B���^�␳�t���e�@�j
	//	���g���̈�ł̐ς���ԗ̈�ł̏􍞂݉��Z�Ƃ��ċL�q�\�Ȃ��Ƃ𗘗p����D
	//	�t�[���G�ϊ��@�ɂ����铊�e���̂P�����t�[���G�ϊ��́C���e���ɍ��拭���t�B���^|s|����ݍ��މ��Z��
	//	���w�I�ɓ��l�Ȃ̂ŁC�t�B���^|s|����ݍ��񂾓��e���𕽖�xy��Ŋp�xth�̔g�Ƃ���th�Őϕ������
	//	���摜������xy�ɕ��������D
	//---------------------------------------------------------------------------------------------
	//	G(s, th)�ɍ��拭���t�B���^�������ċt�ϊ�
	Mat complexDFTImage_rth_s = complexDFTImage_rth.clone();
	for (int i = 0; i < complexDFTImage_rth_s.cols; i++) {
		double d = abs(i - complexDFTImage_rth_s.cols/2);
		complexDFTImage_rth_s.col(i) *= d;
	}
	Mat complexDFTPlanes_rth_s[]
		= { Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1), Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1) };
	Mat magImage_rth_s;
	split(complexDFTImage_rth_s, complexDFTPlanes_rth_s);
	magnitude(complexDFTPlanes_rth_s[0], complexDFTPlanes_rth_s[1], magImage_rth_s);
	magImage_rth_s += Scalar::all(1);
	log(magImage_rth_s, magImage_rth_s);
	normalize(complexDFTPlanes_rth_s[0], magImage_rth_s, 0, 1, CV_MINMAX);
	imshow("X�����e��1����FFT+���拭���t�B���^", magImage_rth_s);
	waitKey();
	//	�tDFT
	cvutil::fftShift1D(complexDFTImage_rth_s, complexDFTImage_rth_s);
	dft(complexDFTImage_rth_s, complexDFTImage_rth_s, DFT_INVERSE + DFT_SCALE + DFT_ROWS);
	//	DFT���ʕ\��
	split(complexDFTImage_rth_s, complexDFTPlanes_rth_s);
	Mat magImage_rth_s_inv;
	normalize(complexDFTPlanes_rth_s[0], magImage_rth_s_inv, 0, 1, CV_MINMAX);
	imshow("X�����e��1����FFT+���拭���t�B���^", magImage_rth_s_inv);

	//	�ώZ
	Mat invDFTImage_filter = Mat::zeros(Size(projectedImage.cols,projectedImage.cols), CV_64FC1);
	for (int th = 0; th < div_rotation; th++) {
		double theta = th * CV_PI / div_rotation;		//	[0, PI]
		Point2d center(invDFTImage_filter.cols / 2 - 0.5, invDFTImage_filter.rows / 2 - 0.5);
		for (int i = 0; i < invDFTImage_filter.rows; i++) {
			for (int j = 0; j < invDFTImage_filter.cols; j++) {
				Point2d p(j - center.x, -i + center.y);
				//	r���̖@�������ɓ����l��������
				double r = p.x*cos(theta) + p.y*sin(theta);
				if (r < -center.x)r = -center.x; else if (r >= center.x) r = center.x;
				invDFTImage_filter.at<double>(i, j) += complexDFTPlanes_rth_s[0].at<double>(th, (int)(r+center.x));
			}
		}
		Mat invDFTImage_filter_scaled;
		normalize(invDFTImage_filter, invDFTImage_filter_scaled, 0, 255, CV_MINMAX);
		invDFTImage_filter_scaled.convertTo(invDFTImage_filter_scaled, CV_8U);
		imshow("���拭���t�B���^�̌���", invDFTImage_filter_scaled);
		cout << "reconstructing... : " << th << " / " << div_rotation << "\r";
		waitKey(1);
	}
	cout << "reconstruction finished!" << endl;
	normalize(invDFTImage_filter, invDFTImage_filter, 0, 255, CV_MINMAX);
	invDFTImage_filter.convertTo(invDFTImage_filter, CV_8U);
	imwrite("s_filter_reconstruction.png", invDFTImage_filter);
	waitKey();
	
	return 0;
}
