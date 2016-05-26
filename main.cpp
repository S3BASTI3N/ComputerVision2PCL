//
// Created by sebastien on 23-5-16.
//
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

#include <pcl/octree/octree.h>
#include <pcl/octree/octree_pointcloud_adjacency.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/surface/texture_mapping.h>
#include <pcl/surface/impl/texture_mapping.hpp>

#include <pcl/features/integral_image_normal.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

// Depth to point cloud
const float MAX_DEPTH = 1.0;

// Poisson
const float SCALE = 1.25;
const int   DEPTH = 10;
const float SAMPLES_PER_NODE = 14;

// Texture
const float RESOLUTION = 0.01f;

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

void mergePointClouds( std::vector<Frame3D> frames, pcl::PointCloud<pcl::PointNormal> &mergedCloud )
{
    for( auto frame : frames )
    {
        // Create pointcloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = Mat2IntegralPointCloud(frame.depth_image_, (float)frame.focal_length_, MAX_DEPTH );

        // Calculate normals
        pcl::PointCloud<pcl::PointNormal>::Ptr normals = computeNormals(cloudPtr);

        // Transform normals
        pcl::PointCloud<pcl::PointNormal>::Ptr transformedNormals(new pcl::PointCloud<pcl::PointNormal>());;
        auto transform = frame.getEigenTransform();

        pcl::transformPointCloudWithNormals( *normals, *transformedNormals, transform );

        std::vector<int> indices;
        pcl::removeNaNNormalsFromPointCloud(*transformedNormals, *transformedNormals, indices );

        mergedCloud += *transformedNormals;
    }
}

pcl::PolygonMesh constructMesh( pcl::PointCloud<pcl::PointNormal>::Ptr cloudPtr )
{
    pcl::Poisson<pcl::PointNormal> poisson;

    poisson.setInputCloud(cloudPtr);

    poisson.setDepth(DEPTH);
    poisson.setScale(SCALE);
    poisson.setSamplesPerNode(SAMPLES_PER_NODE);

    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    return mesh;
}

void visualiseMesh( pcl::PolygonMesh &mesh )
{
    pcl::visualization::PCLVisualizer viewer ("Simple Cloud Viewer");

    viewer.setBackgroundColor (0.1, 0.1, 0.1);
    viewer.addPolygonMesh(mesh,"meshes",0);
    viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters ();
    viewer.setCameraPosition(-1, 1, 1, 0, 0, 0 );
    while (!viewer.wasStopped ()){
        viewer.spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}

Eigen::Matrix4f getInverseCameraMatrix( Eigen::Matrix4f &cameraPose )
{
    auto rotation = cameraPose.block(0, 0, 3, 3);
    auto translation = cameraPose.block(0, 3, 4, 1);

    auto rotationInv = (Eigen::Matrix4f)rotation.inverse();
    auto translationInv = -rotationInv * translation;

    Eigen::Matrix4f cameraPoseInv;
    cameraPoseInv << rotationInv(0,0) ,rotationInv(0,1) ,rotationInv(0,2)   ,translationInv(0)
            ,rotationInv(1,0) ,rotationInv(1,1) ,rotationInv(1,2)   ,translationInv(1)
            ,rotationInv(2,0) ,rotationInv(2,1) ,rotationInv(2,2)   ,translationInv(2)
            ,0                ,0                ,0                  ,1 ;

    return cameraPoseInv;
}

void addTexture(pcl::PolygonMesh &mesh, std::vector<Frame3D> &frames )
{

    auto polygons = mesh.polygons;
    auto cloud = pcl::PointCloud<pcl::PointXYZ>();
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);

    pcl::TextureMapping<pcl::PointXYZ> mapping;

    pcl::PointCloud<pcl::PointXYZRGB> coloredCloud;
    pcl::copyPointCloud(cloud, coloredCloud);

    cv::Vec3d color;
    color[0] = 255;   // b
    color[1] = 0;   // g
    color[2] = 0; // r

    int id = 0;
    for( auto frame : frames )
    {
        //auto depthImage = frame.depth_image_;
        auto focalLength = frame.focal_length_;
        auto cameraPose = frame.getEigenTransform();
        pcl::PointCloud<pcl::PointXYZ> transformedCloud = pcl::PointCloud<pcl::PointXYZ>();

        pcl::TextureMapping<pcl::PointXYZ>::Camera camera;
        camera.focal_length = focalLength*3.8;
        camera.pose = cameraPose;
        camera.width = frame.rgb_image_.cols;
        camera.height = frame.rgb_image_.rows;

        Eigen::Matrix4f cameraPoseInv = getInverseCameraMatrix( cameraPose );
        pcl::transformPointCloud(cloud, transformedCloud, cameraPoseInv);

        //pcl::TextureMapping<pcl::PointXYZ>::Octree tree(RESOLUTION);
        pcl::TextureMapping<pcl::PointXYZ>::Octree::Ptr tree( new pcl::TextureMapping<pcl::PointXYZ>::Octree(RESOLUTION));
        tree->setInputCloud( transformedCloud.makeShared() );
        tree->addPointsFromInputCloud();

        pcl::PointXYZ cameraPoint;
        cameraPoint.x = cameraPoseInv(0,3);
        cameraPoint.y = cameraPoseInv(1,3);
        cameraPoint.z = cameraPoseInv(2,3);

        cv::Mat roi = frame.rgb_image_.clone();

        for( auto polygon : polygons ) {
            for (auto i : polygon.vertices) {
                pcl::PointXYZ point = transformedCloud.points.at(i);

                if( !mapping.isPointOccluded( point, tree ) )
                {
                    Eigen::Vector2f coordinates;
                    if (mapping.getPointUVCoordinates(point, camera, coordinates)) {
                        int x = (int)(coordinates[0] * camera.width);
                        int y = (int)(camera.height - (coordinates[1] * camera.height));

                        cv::Vec3b pixel = frame.rgb_image_.at<cv::Vec3b>(cv::Point(x, y));
                        roi.at<cv::Vec3b>(cv::Point(x, y)) = color;

                        int b = pixel[0];
                        int g = pixel[1];
                        int r = pixel[2];

                        auto coloredPoint = coloredCloud.points.at(i);
                        if( coloredPoint.r != 0 || coloredPoint.g != 0 || coloredPoint.b != 0 )
                        {
                            coloredPoint.r = (coloredPoint.r + (uint8_t)r)/(uint8_t)2;
                            coloredPoint.g = (coloredPoint.g + (uint8_t)g)/(uint8_t)2;
                            coloredPoint.b = (coloredPoint.b + (uint8_t)b)/(uint8_t)2;
                        }
                        else
                        {
                            coloredPoint.r = (uint8_t)r;
                            coloredPoint.g = (uint8_t)g;
                            coloredPoint.b = (uint8_t)b;
                        }

                        coloredCloud.points.at(i) = coloredPoint;
                    }
                }
            }
        }
        cv::imwrite("../roi_" + std::to_string(id) + ".jpg", roi );
        ++id;
    }
    pcl::toPCLPointCloud2(coloredCloud, mesh.cloud );
}

int main()
{
    // Load frames
    std::cout << "Loading frames" << std::endl;
    auto frames = Frame3D::loadFrames("../3dframes/");

    // Merge clouds
    std::cout << "Merge clouds" << std::endl;
    pcl::PointCloud<pcl::PointNormal> mergedCloud;
    pcl::PointCloud<pcl::PointNormal>::Ptr mergedCloudPtr(&mergedCloud);
    mergePointClouds(frames, mergedCloud);

    // Construct mesh
    std::cout << "Construct Mesh" << std::endl;
    auto mesh = constructMesh (mergedCloudPtr);

    // Add texture to point cloud
    std::cout << "Add texture" << std::endl;
    addTexture(mesh, frames);

    // Visualize mesh
    std::cout << "Visualise mesh" << std::endl;
    visualiseMesh(mesh);

    return 1;
}