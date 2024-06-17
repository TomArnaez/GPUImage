#pragma once

#include <image.hpp>

namespace ko::io {
  ko::image::image_2d<uint16_t> read_image(std::string path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_UNCHANGED);
    view<uint16_t**> data("Test", mat.cols, mat.rows);
    auto host_mirror = Kokkos::create_mirror_view(data);
    std::memcpy(host_mirror.data(), mat.data, data.size() * sizeof(uint16_t));
    Kokkos::deep_copy(data, host_mirror);
    ko::image::image_2d<uint16_t> image(data);
    return image;
  }

  void save_image(ko::image::image_2d<uint16_t> img, std::string filepath) {
    cv::Mat cv_img(img.height(), img.width(), CV_16UC1);
    auto host_mirror = Kokkos::create_mirror_view(img.data());
    Kokkos::deep_copy(host_mirror, img.data());
    std::memcpy(cv_img.data, host_mirror.data(), img.element_count() * sizeof(uint16_t));
    if (!cv::imwrite(filepath, cv_img)) {
      throw std::runtime_error("Failed to save image: " + filepath);
    }
  }
}