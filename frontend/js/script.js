const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

imageInput.addEventListener("change", function () {

const file = imageInput.files[0];

if(file){
const reader = new FileReader();

reader.onload = function(e){
preview.src = e.target.result;
preview.style.display = "block";
}

reader.readAsDataURL(file);
}

});

async function predictImage(){

const file = imageInput.files[0];

if(!file){
alert("Please upload an image first!");
return;
}

const formData = new FormData();
formData.append("file", file);

result.innerText = "Predicting...";

try{

const response = await fetch("http://127.0.0.1:5000/predict",{
method:"POST",
body:formData
});

const data = await response.json();

if(data.prediction === "Malignant"){
result.style.color = "red";
}
else{
result.style.color = "lightgreen";
}

result.innerText = "Prediction: " + data.prediction;

}catch(error){

result.innerText = "Server error. Make sure backend is running.";

}

}