
# Streamlit Machine Learning Application

This Streamlit application allows users to choose from several machine learning models, explore and analyze their performance metrics, and visualize the confusion matrix to evaluate their ability to make accurate predictions.

## Getting Started

Make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn plotly
```

## Running the Application

To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```



## Features

- **Home Page:**
  - Displays a welcome message introducing the application.

- **Model Page:**
  - Allows users to choose from various machine learning models, including Random Forest, SVM, Naive Bayes, Ridge, and Logistic Regression.
  - Provides sliders for adjusting hyperparameters specific to each selected model.
  - Evaluates and displays model performance metrics, such as accuracy and a confusion matrix heatmap.
  - Allows users to make predictions for a selected model based on input sliders.

## Dependencies

- [Streamlit](https://www.streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Plotly Express](https://plotly.com/)

## Usage

1. Run the Streamlit application using the provided command.
2. Choose the "Model" option from the sidebar.
3. Select a machine learning model from the dropdown menu.
4. Adjust hyperparameters using the sliders (if applicable).
5. Explore the model performance metrics and confusion matrix.
6. Make predictions based on user input.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

## Application Link

To access the deployed application [click here](https://extractiondeconnaissances-gudaijqninhvq9zjvt2niw.streamlit.app/).
