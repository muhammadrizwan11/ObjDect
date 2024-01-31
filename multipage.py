import streamlit as st
from PIL import Image

# Function for Object Detection (Replace this with your actual object detection code)
def object_detection():
    
    # Your object detection code here
    
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.utils import draw_bounding_boxes
    import zipfile
    import os

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    categories = weights.meta["categories"]
    img_preprocess = weights.transforms()

    # Function to load the model
    @st.cache_resource
    def load_model():
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        model.eval()
        return model

    model = load_model()

    # Function to make predictions
    def make_prediction(img):
        img_processed = img_preprocess(img)
        prediction = model(img_processed.unsqueeze(0))[0]
        prediction["labels"] = [categories[label] for label in prediction["labels"]]
        return prediction

    # Function to create image with bounding boxes
    def create_image_with_bboxes(img, prediction):
        img_tensor = torch.tensor(img)
        img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"],
                                          labels=prediction["labels"],
                                          colors=["red" if label == "person" else "green" for label in
                                                  prediction["labels"]], width=2)
        img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
        return img_with_bboxes_np

    # Function to handle dataset upload and extract images
    def handle_dataset_upload(zip_file):
        images = []

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract the contents of the ZIP file to a temporary directory
            temp_dir = "temp_dataset"
            zip_ref.extractall(temp_dir)

            # Get a list of image files in the extracted directory
            image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Load each image and append to the images list
            for image_file in image_files:
                image_path = os.path.join(temp_dir, image_file)
                image = Image.open(image_path)
                images.append(image)

        return images

    # Dashboard
    st.title("Object Detector :tea: :coffee:")

    # Step 1: Upload ZIP Dataset
    upload_dataset = st.file_uploader(label="Upload Dataset (ZIP file format):", type=["zip"])

    if upload_dataset:
        # Handle dataset upload and extract images
        images = handle_dataset_upload(upload_dataset)

        # Display the extracted images and predictions
        for image in images:
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make predictions and display bounding boxes
            prediction = make_prediction(image)
            img_with_bbox = create_image_with_bboxes(np.array(image).transpose(2, 0, 1), prediction)

            # Display the image with bounding boxes
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            plt.imshow(img_with_bbox)
            plt.xticks([], [])
            plt.yticks([], [])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)

            # Display predicted probabilities
            del prediction["boxes"]
            st.header("Predicted Probabilities")
            st.write(prediction, style={"labels": {"font-size": "20px"}})

        
        # Button to export the trained model
        if st.button("Export Trained Model"):
            # Save the model to a file
            model_filename = "trained_model.pth"
            torch.save(model.state_dict(), model_filename)
            st.success(f"Trained model exported successfully as {model_filename}")


    

# Function for Image Classification (Replace this with your actual image classification code)
def image_classifier():
    st.title("Image Classifier Page")
    # Your image classification code here
#from pyexpat import model
    
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.models import resnet50,ResNet50_Weights
    from captum.attr import IntegratedGradients
    from captum.attr import visualization as viz


    preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
    categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta['categories'])

    @st.cache_resource
    def load_model():
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        return model

    def make_prediction(model, processed_img):

        probs = model (processed_img.unsqueeze(0))
        probs = probs.softmax(1)
        probs = probs [0] .detach().numpy()
        prob, idxs = probs[probs.argsort()[-5:][::-1]],probs.argsort()[-5:][::-1]
        return  prob, idxs

    def interpret_prediction(model, processed_img, target):
        interpretation_algo = IntegratedGradients(model)
        feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
        feature_imp = feature_imp[0].numpy()
        feature_imp = feature_imp.transpose(1,2,0)
        return feature_imp
    #Dashbord
    st.title("Image Classsifire :tea: :coffee:")
    upload = st.file_uploader(label="Upload image:" , type=["png","jpg" , "jpeg"])
    if upload:
    
        img  = Image.open(upload)
        model = load_model()
        processed_img = preprocess_func(img)
        probs , idxs = make_prediction(model , processed_img)
        feature_imp = interpret_prediction(model,processed_img , idxs[0])
        main_fig  = plt.figure(figsize=(12,3))
        ax = main_fig.add_subplot(111)
        plt.barh(y= categories[idxs][::-1] , width=probs[::-1] , color = ["dodgerblue"]*4 +"tomato")
        plt.title("Top 5 Possibilities" , loc = "center" , fontsize = 15)
        st.pyplot(main_fig , use_container_width=True)
        inter_fig , ax = viz.visualize_image_attr(feature_imp , show_colorbar = True , fig_size=(6,6))
        col1 , col2 =st.columns(2,gap="medium")
        with col1:
            main_fig = plt.figure(figsize=(6,6))
            ax =main_fig.add_subplot(111)
            plt.imshow(img)
            plt.xticks([],[])
            plt.yticks([],[])
            st.pyplot(main_fig ,use_container_width=True)
        with col2:
            st.pyplot(inter_fig, use_container_width=True)



# Function for Creating Graph (Replace this with your actual graph creation code)
def create_graph():
    st.title("Create Graph Page")
    # Your graph creation code here

# Main Streamlit App
def main():
    st.title("Multi-Page Streamlit App")

    # Sidebar with navigation buttons
    selected_page = st.sidebar.selectbox("Select Page", ["Home", "Object Detection", "Image Classifier", "Create Graph"])

    # Page content based on selection
    if selected_page == "Home":
        st.write("Welcome to the Home Page")
    elif selected_page == "Object Detection":
        object_detection()
    elif selected_page == "Image Classifier":
        image_classifier()
    elif selected_page == "Create Graph":
        create_graph()

# Run the app
if __name__ == "__main__":
    main()
