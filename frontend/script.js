document.getElementById("sharpenBtn").addEventListener("click", upload);

function upload(event) {
  if (event) event.preventDefault();

  const input = document.getElementById("imgInput");

  if (!input.files[0]) {
    alert("⚠️ Please choose an image before clicking Sharpen.");
    return;
  }

  const formData = new FormData();
  formData.append("image", input.files[0]);

  const originalURL = URL.createObjectURL(input.files[0]);

  fetch("http://127.0.0.1:5000/sharpen", {
    method: "POST",
    body: formData,
  })
    .then(res => res.json())
    .then(data => {
      const resultDiv = document.getElementById("result");
      const ts = Date.now(); // Prevent caching in Live Server

      resultDiv.innerHTML = `
        <h3>SSIM Score: ${data.ssim.toFixed(4)}</h3>
        <div style="display: flex; gap: 20px;">
          <div>
            <p><strong>Original</strong></p>
            <img src="${originalURL}" width="300" />
          </div>
          <div>
            <p><strong>Sharpened</strong></p>
            <img src="http://127.0.0.1:5000/preview/${data.output_url.split('/').pop()}?t=${ts}" width="300" />
          </div>
        </div>
      `;
    })
    .catch((err) => {
      console.error("Fetch error:", err);
      alert("Something went wrong. Check if Flask is running.");
    });
}
