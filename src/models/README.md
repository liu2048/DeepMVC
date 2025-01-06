# Models Directory

This directory contains the implementation of various models used in the project. Each model is implemented in its own subdirectory, and the corresponding loss functions are also included.

## Subdirectories and Files

- `comvc`: Contains the implementation of the CoMVC model and its loss functions.
- `contrastive_ae`: Contains the implementation of the Contrastive Autoencoder model.
- `dmsc`: Contains the implementation of the DMSC model and its loss functions.
- `eamc`: Contains the implementation of the EAMC model and its loss functions.
- `mimvc`: Contains the implementation of the MIMVC model and its loss functions.
- `mvae`: Contains the implementation of the MVAE model and its loss functions.
- `mviic`: Contains the implementation of the MvIIC model and its loss functions.
- `mvscn`: Contains the implementation of the MVSCN model and its loss functions.
- `simvc`: Contains the implementation of the SiMVC model.

For detailed descriptions of the files in each subdirectory, please refer to the respective `README.md` files in the corresponding subdirectories.

## Toggle Chinese Translation

To view the Chinese translation of this document, click the button below:

<button onclick="toggleTranslation()">Toggle Chinese Translation</button>

<script>
function toggleTranslation() {
  var x = document.getElementById("chinese-translation");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}
</script>

<div id="chinese-translation" style="display:none;">
  <h1>模型目录</h1>
  <p>此目录包含项目中使用的各种模型的实现。每个模型都在其自己的子目录中实现，并且还包括相应的损失函数。</p>

  <h2>子目录和文件</h2>
  <ul>
    <li><code>comvc</code>：包含CoMVC模型及其损失函数的实现。</li>
    <li><code>contrastive_ae</code>：包含对比自编码器模型的实现。</li>
    <li><code>dmsc</code>：包含DMSC模型及其损失函数的实现。</li>
    <li><code>eamc</code>：包含EAMC模型及其损失函数的实现。</li>
    <li><code>mimvc</code>：包含MIMVC模型及其损失函数的实现。</li>
    <li><code>mvae</code>：包含MVAE模型及其损失函数的实现。</li>
    <li><code>mviic</code>：包含MvIIC模型及其损失函数的实现。</li>
    <li><code>mvscn</code>：包含MVSCN模型及其损失函数的实现。</li>
    <li><code>simvc</code>：包含SiMVC模型的实现。</li>
  </ul>

  <p>有关每个子目录中文件的详细描述，请参阅相应子目录中的<code>README.md</code>文件。</p>
</div>
