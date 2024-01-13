import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

def run_clarifai_inference(pat, user_id, app_id, model_id, model_version_id, image_url):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + pat),)

    user_data_object = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=user_data_object,
            model_id=model_id,
            version_id=model_version_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            url=image_url
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        st.error(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    predicted_concepts = [(concept.name, concept.value) for concept in output.data.concepts]
    return predicted_concepts

def main():
    st.title("Clarifai Inference Streamlit App")

    pat = st.text_input("Enter your Personal Access Token (PAT):")
    user_id = st.text_input("Enter your User ID:")
    app_id = st.text_input("Enter your App ID:")
    model_id = st.text_input("Enter your Model ID:")
    model_version_id = st.text_input("Enter your Model Version ID:")
    image_url = st.text_input("Enter the Image URL:")

    if st.button("Run Clarifai Inference"):
        if pat and user_id and app_id and model_id and model_version_id and image_url:
            try:
                predicted_concepts = run_clarifai_inference(pat, user_id, app_id, model_id, model_version_id, image_url)
                st.success("Predicted concepts:")
                for concept_name, concept_value in predicted_concepts:
                    st.write(f"{concept_name}: {concept_value:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please fill in all the required fields.")

if __name__ == "__main__":
    main()
