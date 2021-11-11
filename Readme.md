
## Task

1. Load specific Tensorflow model (DenseNet121)
2. Set up a fastAPI to make predictions when provided with an image via a POST request
3. Change port to 5000
4. respond with {"response": "cat"}

`conda create --name fastapi-ml`

`conda activate fastapi-ml`

`pip install -r requirements.txt`

To run this API use the following command
`uvicorn main:app --reload --port 5000`


### Areas for development
1. flexibility of models to load
2. basic CI/CD using Github to test
3. Write tests (pytest preferred), I think there are some fastAPI tools I havent explored yet
4. Provide it in a container for reproducibility (begun drafting but havent finished)
5. I think a conda yaml would have been neater than using requirements.txt with a bit more time
6. Specify version numbers for packages.

### References
- https://detailed-tutorials.sthakur.work/all-tutorials/using-pretrained-models/
- https://towardsdatascience.com/image-classification-api-with-tensorflow-and-fastapi-fc85dc6d39e8
- https://testdriven.io/blog/fastapi-machine-learning/
