namespace ko::shapes {
    struct rect_2d {
        size_t width_;
        size_t height_;

        rect_2d(): width_(0), height_(0) {}
        rect_2d(size_t width, size_t height) : width_(width), height_(height) {}
    };
}