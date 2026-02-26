def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    grayscale = [[0 for _ in range(len(image[0]))] for _ in range(len(image))]

    for i in range(len(image)):
        for j in range(len(image[i])):
            grayscale[i][j] = 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]

    return grayscale