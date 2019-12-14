#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
//#include "OpticalFlow.h"

#include <iostream>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <time.h>

//#define DEBUG_PRINT

int main(int argc, char** argv)
{
	VideoCapture capture;
	char* video = argv[1];
	int flag = arg_parse(argc, argv);
	//-- 1. Load the cascades
    //if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);
#if 0
	std::vector<Frame> bb_list;
	if(bb_file) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}
#endif
	if(flag)
		seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track)
		namedWindow("DenseTrackStab", 0);

	/******************** CUDA VARIABLES ********************/
	// images
	cuda::GpuMat gpu_prev_grey, gpu_grey, gpu_warped;

	// descriptors
	cuda::GpuMat prev_desc_surf, desc_surf;

	// optical flow
	cuda::GpuMat gpu_flow, gpu_flow_warp;

	// SURF Detector
	cuda::SURF_CUDA surf = cuda::SURF_CUDA(200);

	// Corner Detector default threshold is 10
	Ptr< cuda::FastFeatureDetector > fastDetector = cuda::FastFeatureDetector::create();

	// Point Matcher
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();

	// Optical Flow Calculator
	Ptr<cuda::FarnebackOpticalFlow> flowObj = cuda::FarnebackOpticalFlow::create();
	flowObj->setFastPyramids(true);
	/******************** CUDA VARIABLES ********************/

	Mat image, prev_grey, grey, grey_pyr, flow_pyr, flow_warp_pyr;

	// Feature Points
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;	// keypoints
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<KeyPoint> tracking_points;

	std::vector< std::vector< DMatch> > matches;
	std::list<Track> tracks;
	int init_counter = 0; // indicate when to detect new feature points
	
	while(true) {
		Mat frame;
		int i, c;
		Frame bb_frame = Frame(frame_num);

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		if(frame_num == start_frame) {
			//fprintf(stderr, "video dimensions: %ix%i\n", frame.rows, frame.cols);
			
			//determine a suitable number of pyramid layers based on frame size
			scale_num = CalculatePyrLevels(frame);
			flowObj->setNumLevels(scale_num);
#ifdef DEBUG_PRINT
			std::cout << "NUM LEVELS: " << scale_num << std::endl;
#endif
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(image);
			cvtColor(image, prev_grey, COLOR_BGR2GRAY);

			// UPLOAD to GPU memory
			gpu_prev_grey = cuda::GpuMat( prev_grey );

			//-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
			surf( gpu_prev_grey, cuda::GpuMat(), prev_kpts_surf, prev_desc_surf );
			fastDetector->detect(gpu_prev_grey, tracking_points);
#ifdef DEBUG_PRINT
			std::cout << "FIRST FRAME: " << prev_kpts_surf.size() << " SURF keypoints" << std::endl;
			std::cout << "FIRST FRAME: " << tracking_points.size() << " Corner keypoints" << std::endl;
#endif
			// save the feature points
			for(i = 0; i < tracking_points.size(); i++)
				tracks.push_back(Track(tracking_points[i].pt, trackInfo, hogInfo, hofInfo, mbhInfo));

			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, COLOR_BGR2GRAY);

		// UPLOAD to GPU memory
		gpu_grey = cuda::GpuMat(grey);

		// SURF
		surf( gpu_grey, cuda::GpuMat(), kpts_surf, desc_surf );
#ifdef DEBUG_PRINT
		std::cout << "FRAME " << frame_num << ": " << prev_kpts_surf.size() << " SURF keypoints" << std::endl;
#endif
		// Matching descriptor vectors using BruteForceMatcher
		matcher->knnMatch(prev_desc_surf, desc_surf, matches, 2);

		for (int i = 0; i < std::min(kpts_surf.size()-1, matches.size()); i++)
		{
			//if ((matches[k][0].distance < 0.6*(matches[k][1].distance))
			//&& ((int)matches[k].size() <= 2 && (int)matches[k].size()>0))
			{
				// get the point pairs that are successfully matched
				int j = matches[i][0].queryIdx;
				int k = matches[i][0].trainIdx;
				prev_pts_surf.push_back(prev_kpts_surf[j].pt);
				pts_surf.push_back(kpts_surf[k].pt);
			}
		}
#ifdef DEBUG_PRINT
		std::cout << prev_pts_surf.size() << " SURF matches" << std::endl;

//		for(int i = 0; i < pts_surf.size(); i++)
//		{
//			circle(image, pts_surf[i], 2, Scalar(0,255,0), -1, 8, 0);
//		}
#endif
		// HOMOGRAPHY
		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_surf.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_surf, pts_surf, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}
#ifdef DEBUG_PRINT
		std::cout << "H = "<< std::endl << " "  << H << std::endl << std::endl;
#endif
		// WARP FRAME
		cuda::warpPerspective( gpu_grey, gpu_warped, H, gpu_grey.size());

		// OPTICAL FLOW
		flowObj->calc(gpu_prev_grey, gpu_grey, gpu_flow);
		flowObj->calc(gpu_prev_grey, gpu_warped, gpu_flow_warp);

		// DOWNLOAD from GPU memory
		gpu_flow.download(flow_pyr);
		gpu_flow_warp.download(flow_warp_pyr);

		int width = grey.cols;
		int height = grey.rows;

		DescMat* hogMat = NULL;
		DescMat* hofMat = NULL;
		DescMat* mbhMatX = NULL;
		DescMat* mbhMatY = NULL;

		// compute the integral histograms
		if(HOG_flag) {
			hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(grey, hogMat->desc, hogInfo);
		}

		if(HOF_flag) {
			hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_warp_pyr, hofMat->desc, hofInfo);
		}

		if(MBH_flag) {
			mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_warp_pyr, mbhMatX->desc, mbhMatY->desc, mbhInfo);
		}

		// track feature points
		for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
			int index = iTrack->index;
			Point2f prev_point = iTrack->point[index];
			int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
			int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

			Point2f point;
			point.x = prev_point.x + flow_pyr.ptr<float>(y)[2*x];
			point.y = prev_point.y + flow_pyr.ptr<float>(y)[2*x+1];

			if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
				iTrack = tracks.erase(iTrack);
				continue;
			}

			iTrack->disp[index].x = flow_warp_pyr.ptr<float>(y)[2*x];
			iTrack->disp[index].y = flow_warp_pyr.ptr<float>(y)[2*x+1];

			// get the descriptors for the feature point
			RectInfo rect;
			GetRect(prev_point, rect, width, height, hogInfo);
			if(HOG_flag) { GetDesc(hogMat, rect, hogInfo, iTrack->hog, index); }
			if(HOF_flag) { GetDesc(hofMat, rect, hofInfo, iTrack->hof, index); }
			if(MBH_flag) {
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
			}
			iTrack->addPoint(point);
#if 0
			if(show_track) {
				//action trajectory
				DrawTrack(iTrack->point, iTrack->index, 1, image, RED);
			}
#endif
			//another idea is to simply capture all trajectories once "gap" frames have passed
			//if(init_counter >= trackInfo.gap) {

			// if the trajectory achieves the maximal length
			if(iTrack->index >= trackInfo.length) {
				std::vector<Point2f> trajectory(trackInfo.length+1);
				for(int i = 0; i <= trackInfo.length; ++i)
					trajectory[i] = iTrack->point[i];

				std::vector<Point2f> displacement(trackInfo.length);
				for (int i = 0; i < trackInfo.length; ++i)
					displacement[i] = iTrack->disp[i];

				float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
				if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) == true) {
					bool actionTrajectory = IsCameraMotion(displacement) ? false : true;
					if(actionTrajectory && verbose_flag) {
						if(HOG_flag) { PrintDesc(iTrack->hog, hogInfo, trackInfo); }
						if(HOF_flag) { PrintDesc(iTrack->hof, hofInfo, trackInfo); }
						if(MBH_flag) {
							PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
							PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
						}
						printf("\n");
					}
//#if 0
					if(show_track) {
						if(actionTrajectory) {
							//action trajectory
							DrawTrack(iTrack->point, iTrack->index, 1, image, GREEN);
						}
						else {
							//camera motion trajectory
							DrawTrack(iTrack->point, iTrack->index, 1, image, WHITE);
						}
					}
//#endif
				}

				iTrack = tracks.erase(iTrack);
				continue;
			}
			++iTrack;
		}
		ReleDescMat(hogMat);
		ReleDescMat(hofMat);
		ReleDescMat(mbhMatX);
		ReleDescMat(mbhMatY);

		// free GPU memory
		gpu_warped.release();
		gpu_flow.release();
		gpu_flow_warp.release();

		// detect new feature points every gap frames
		if(init_counter >= trackInfo.gap) {
			tracking_points.clear();
			fastDetector->detect(gpu_grey, tracking_points);
#ifdef DEBUG_PRINT
			std::cout << "Detected: " << tracking_points.size() << " NEW Corner keypoints" << std::endl;
#endif
			// save the feature points
			for(i = 0; i < tracking_points.size(); i++)
				tracks.push_back(Track(tracking_points[i].pt, trackInfo, hogInfo, hofInfo, mbhInfo));

			init_counter = 0;
		}

		grey.copyTo(prev_grey);
		prev_kpts_surf = kpts_surf;

		prev_desc_surf.release();
		gpu_prev_grey.release();
		desc_surf.copyTo(prev_desc_surf);
		gpu_grey.copyTo(gpu_prev_grey);

		frame_num++;

		if(show_track) {
			imshow( "DenseTrackStab", image);
			c = waitKey(3);
			if((char)c == 27) break;
		}
	}

	if(show_track)
		destroyWindow("DenseTrackStab");

	return 0;
}
