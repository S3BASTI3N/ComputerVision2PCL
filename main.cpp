//
// Created by sebastien on 23-5-16.
//

const float MAX_DEPTH = 1.0;


#include "Frame3D/Frame3D.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/gp3.h>

#include <pcl/filters/filter.h>
#include <pcl/surface/mls.h>

#include <pcl/features/integral_image_normal.h>

#include <opencv2/core/eigen.hpp>


pcl::PointCloud<pcl::PointXYZ>::Ptr Mat2IntegralPointCloud( const cv::Mat& depth_mat, const float focal_length, const float max_depth)
{
    assert(depth_mat.type() == CV_16U);
    pcl::PointCloud<pcl::PointXYZ> ::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ> ());
    const int half_width = depth_mat.cols / 2;
    const int half_height = depth_mat.rows / 2;
    const float inv_focal_length = 1.0f / focal_length;
    point_cloud->points.reserve( depth_mat.rows * depth_mat.cols);
    for (int y = 0; y < depth_mat.rows; y++) {
        for (int x = 0; x < depth_mat.cols; x++) {
            float z = depth_mat.at<ushort>(cv:: Point(x, y)) * 0.001f;
            if (z < max_depth && z > 0) {
                point_cloud->points.emplace_back(static_cast<float>(x - half_width)  * z * inv_focal_length,
                                                  static_cast<float>(y - half_height) * z * inv_focal_length,
                                                  z);
            } else {
                point_cloud->points.emplace_back(x,y,NAN);
            }
        }
    }
    point_cloud->width = (uint32_t)depth_mat.cols;
    point_cloud->height = (uint32_t)depth_mat.rows;
    return point_cloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>); // Output datasets
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    ne.setNormalEstimationMethod( ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*cloud_normals);
    copyPointCloud(*cloud, *cloud_normals);
    return cloud_normals;
}

void mergePointClouds( pcl::PointCloud<pcl::PointNormal> &mergedCloud )
{
    std::string fileName;
    for( int i = 0; i < 7; i++ )
    {

        // Load depth image with info
        fileName = "/home/sebastien/Downloads/FinalAssignment_CV2_2016/3dframes/0000" + std::to_string(i) + ".3df";
        Frame3D frame(fileName);

        // Create pointcloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = Mat2IntegralPointCloud(frame.depth_image_, (float)frame.focal_length_, MAX_DEPTH );

        // Calculate normals
        pcl::PointCloud<pcl::PointNormal>::Ptr normals = computeNormals(cloudPtr);

        // Transform normals
        pcl::PointCloud<pcl::PointNormal>::Ptr transformedNormals(new pcl::PointCloud<pcl::PointNormal>());;
        auto transform = frame.getEigenTransform();


        pcl::transformPointCloudWithNormals( *normals, *transformedNormals, transform );

        mergedCloud += *transformedNormals;
    }
}

int main()
{

    pcl::PointCloud<pcl::PointNormal> mergedCloud;
    mergePointClouds(mergedCloud);
    pcl::PointCloud<pcl::PointNormal>::ConstPtr mergedCloudPtr(&mergedCloud);

    std::cout << mergedCloudPtr->points.size() << std::endl;

    std::vector<int> indices;
    pcl::removeNaNNormalsFromPointCloud(mergedCloud, mergedCloud, indices );

    std::cout << mergedCloudPtr->points.size() << std::endl;

    cout << "begin poisson reconstruction" << endl;
    pcl::Poisson<pcl::PointNormal> poisson;
    //poisson.setSamplesPerNode(3);

    poisson.setDepth(9);
    poisson.setInputCloud(mergedCloudPtr);
    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    pcl::visualization::PCLVisualizer viewer ("Simple Cloud Viewer");

    viewer.setBackgroundColor (0, 0, 0);
    viewer.addPolygonMesh(mesh,"meshes",0);
    viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters ();
    while (!viewer.wasStopped ()){
        viewer.spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return -1;
}