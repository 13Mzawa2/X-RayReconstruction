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

const int div_rotation = 360;						//	360�������ĎB�e
const int projectionRowSize = 820;						//	���e���̕�

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
	Mat projectedImage = Mat(div_rotation, projectionRowSize, CV_64FC1, Scalar::all(0));	//	���e���x�N�g���̏W�� �s�����p�x
	cout << "���e���T�C�Y : " << projectedImage.size() << endl;
	//	��]�ptheta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	�p�xrad�ɕϊ��i�S����180deg�j
		//	�摜����]
		Mat rotatedImage;		//	���̉摜�̒����Ō��摜����]������
		int offsety = projectionRowSize - srcd.rows;
		int offsetx = projectionRowSize - srcd.cols;
		copyMakeBorder(srcd, rotatedImage, offsety / 2, offsety / 2, offsetx / 2, offsetx / 2, BORDER_CONSTANT, Scalar::all(0));//	�摜���S�Ɖ�]���S�����킹��
		Mat affine = getRotationMatrix2D(Point2d(rotatedImage.cols / 2, rotatedImage.rows / 2), -theta / CV_PI * 180, 1.0);		//	�A�t�B���ϊ��s��
		warpAffine(rotatedImage, rotatedImage, affine, rotatedImage.size(), 1, 0, Scalar::all(0));
		//	�摜������X���𓖂Ăĉ摜�㕔�ŎB�e
		Mat reduceRow(Size(rotatedImage.cols, 1), CV_64FC1, Scalar::all(0));
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
	//	3. ���e������̍č\���i�t�B���^�␳�t���e�@�j
	//	���g���̈�ł̐ς���ԗ̈�ł̏􍞂݉��Z�Ƃ��ċL�q�\�Ȃ��Ƃ𗘗p����D
	//	�t�[���G�ϊ��@�ɂ����铊�e���̂P�����t�[���G�ϊ��́C���e���ɍ��拭���t�B���^|s|����ݍ��މ��Z��
	//	���w�I�ɓ��l�Ȃ̂ŁC�t�B���^|s|����ݍ��񂾓��e���𕽖�xy��Ŋp�xth�̔g�Ƃ���th�Őϕ������
	//	���摜������xy�ɕ��������D
	//---------------------------------------------------------------------------------------------

	//	3.1. �B�e�摜g(r, th)��1�����t�[���G�ϊ�����G(s, th)�𓾂�
	Mat optDFTImage_rth = projectedImage.clone();
	//	���f���摜���쐬
	Mat complexDFTPlanes_rth[] = { Mat_<double>(optDFTImage_rth), Mat::zeros(optDFTImage_rth.size(), CV_64F) };
	Mat complexDFTImage_rth;
	merge(complexDFTPlanes_rth, 2, complexDFTImage_rth);
	//	1����DFT���s
	dft(complexDFTImage_rth, complexDFTImage_rth, DFT_ROWS);
	cvutil::fftShift1D(complexDFTImage_rth, complexDFTImage_rth);	//	FFT�摜���]
	//	DFT���ʕ\��
	Mat magImage_rth;
	cvutil::fftMagnitude(complexDFTImage_rth, magImage_rth);
	imshow("X�����e��1����FFT����", magImage_rth);
	//	���ʂ̕ۑ�
	normalize(magImage_rth, magImage_rth, 0, 255, CV_MINMAX);
	magImage_rth.convertTo(magImage_rth, CV_8UC1);
	imwrite("FFT1D_projection_rth.png", magImage_rth);

	//	3.2. G(s, th)�ɍ��拭���t�B���^�������ċt�ϊ�
	Mat complexDFTImage_rth_s = complexDFTImage_rth.clone();
	for (int i = 0; i < complexDFTImage_rth_s.cols; i++) {
		double d = abs(i - complexDFTImage_rth_s.cols / 2 + 0.5);
		complexDFTImage_rth_s.col(i) *= d;
	}
	Mat complexDFTPlanes_rth_s[]
		= { Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1), Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1) };
	//	���ʂ̕\��
	Mat magImage_rth_s;
	cvutil::fftMagnitude(complexDFTImage_rth_s, magImage_rth_s);
	imshow("X�����e��1����FFT+���拭���t�B���^", magImage_rth_s);
	//	�tDFT
	cvutil::fftShift1D(complexDFTImage_rth_s, complexDFTImage_rth_s);
	dft(complexDFTImage_rth_s, complexDFTImage_rth_s, DFT_INVERSE + DFT_SCALE + DFT_ROWS);
	//	DFT���ʕ\��
	split(complexDFTImage_rth_s, complexDFTPlanes_rth_s);
	Mat magImage_rth_s_inv;
	normalize(complexDFTPlanes_rth_s[0], magImage_rth_s_inv, 0, 1, CV_MINMAX);
	imshow("X�����e��1����FFT+���拭���t�B���^�̋tFFT", magImage_rth_s_inv);
	waitKey();

	//	3.3. ��Ԏ��g����ώZ
	Mat invProjectedImage_filter = Mat::zeros(Size(projectionRowSize, projectionRowSize), CV_64FC1);
	for (int th = 0; th < div_rotation; th++) {
		double theta = th * CV_PI / div_rotation;		//	[0, PI]
		Point2d center(invProjectedImage_filter.cols / 2 - 0.5, invProjectedImage_filter.rows / 2 - 0.5);
		//	theta�����̋�Ԏ��g���摜���쐬���ĐώZ
		for (int i = 0; i < invProjectedImage_filter.rows; i++) {
			for (int j = 0; j < invProjectedImage_filter.cols; j++) {
				Point2d p(j - center.x, -i + center.y);
				//	r���̖@�������ɓ����l��������
				double r = p.x*cos(theta) + p.y*sin(theta);
				if (r < -center.x)r = -center.x; else if (r >= center.x) r = center.x;
				invProjectedImage_filter.at<double>(i, j) += complexDFTPlanes_rth_s[0].at<double>(th, (int)(r + center.x));
			}
		}
		Mat invProjectedImage_filter_scaled;
		normalize(invProjectedImage_filter, invProjectedImage_filter_scaled, 0, 255, CV_MINMAX);
		invProjectedImage_filter_scaled.convertTo(invProjectedImage_filter_scaled, CV_8U);
		imshow("���拭���t�B���^�̌���", invProjectedImage_filter_scaled);
		cout << "reconstructing... : " << th << " / " << div_rotation << "\r";
		waitKey(1);
	}
	cout << "reconstruction finished!" << endl;
	normalize(invProjectedImage_filter, invProjectedImage_filter, 0, 255, CV_MINMAX);
	invProjectedImage_filter.convertTo(invProjectedImage_filter, CV_8U);
	imwrite("s_filter_reconstruction.png", invProjectedImage_filter);
	waitKey();

	return 0;
}
