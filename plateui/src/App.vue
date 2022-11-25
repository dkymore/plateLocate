<template>
  <div>
    <h1>Plate UI</h1>
    <h2>by dkymore</h2>
    <el-upload
      ref="upload"
      class="upload-demo"
      drag
      action="http://127.0.0.1:7899"
      :limit="1"
      :before-upload="beforeUpload"
      :on-exceed="handleExceed"
      :auto-upload="false"
    >
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">拖入车牌照片或是<em>点击上传</em></div>
    </el-upload>
    <el-button type="primary" style="margin-top: 30px" @click="delimg"
      >开始识别</el-button
    >
    <h2>车牌预览</h2>
    <el-image
      fit="contain"
      class="im"
      :src="preview"
      style="margin-top: 30px"
    >
    </el-image>
    <h2>车牌检测</h2>
    <el-image
      fit="contain"
      class="imc"
      :src="plateview"
      style="margin-top: 30px"
    >
    </el-image>
    <h2>车牌识别</h2>
    <h3>{{ state.res }}</h3>
  </div>
</template>

<script setup>
import { UploadFilled } from "@element-plus/icons-vue";
import { ElMessageBox, genFileId } from "element-plus";
import axios from "axios";
import { ref } from "vue";
import { ElNotification } from "element-plus";
import { reactive } from "vue";
import { ElLoading } from "element-plus";

const MOCKURL = "http://127.0.0.1:7899";

const preview = ref("");
const plateview = ref("");

const state = reactive({ res: "等待车牌" });

const upload = ref("upload");

var fileobj = null;

const getBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (e) => {
      resolve(reader.result);
    };
    reader.onerror = (error) => reject(error);
  });
};

const handleExceed = (files) => {
  upload.value.clearFiles();
  const file = files[0];
  file.uid = genFileId();
  upload.value.handleStart(file);
  fileobj = file;
};

const beforeUpload = (file) => {
  fileobj = file;
  return false;
};

const submitUpload = () => {
  upload.value.submit();
};

const delimg = async () => {
  try {
    upload.value.submit();
  } catch {}

  if (fileobj === null) {
    return ElNotification({ title: "请先上传图片", type: "error" });
  }
  console.log(fileobj);
  var b64img = null;
  try {
    b64img = await getBase64(fileobj);
    preview.value= b64img;
  } catch (e) {
    ElNotification({ title: "上传失败", type: "error" });
    throw e;
  }

  const loading = ElLoading.service({
    lock: true,
    text: "等待计算中... \n20s超时",
    background: "rgba(0, 0, 0, 0.7)",
  });

  try {
    var res = await axios({
      url: MOCKURL + "/api",
      data: {
        img: b64img,
      },
      params: {
        imgname: fileobj.name,
      },
      method: "post",
      timeout: 20000,
    });
    if (res.data.error) {
      return ElNotification({
        title: "识别失败",
        message: res.data.message,
        type: "error",
      });
    }
    if (res.data.imgdata.length == 0) {
      return ElNotification({
        title: "识别失败",
        message: "没有找到车牌",
        type: "error",
      });
    }
    plateview.value = res.data.imgdata[0].plate;
    state.res = res.data.imgdata[0].res;
  } catch (e) {
    //ElNotification({ title: "获取失败", type: "error" });
    ElMessageBox.alert("解析失败 发生错误：" + e);
    throw e;
  } finally {
    loading.close();
  }

  // axios.post(MOCKURL+"/api",file).then((res)=>{
  //   console.log(res)
  //   imgurl.value.preview = res.data.preview
  //   imgurl.value.plateview = res.data.plateview
  //   res.value = res.data.plate
  // })
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin: auto;
  margin-top: 60px;
  max-width: 600px;
}

.im {
  width: 560px;
  height: 300px;
}

.imc {
  width: 200px;
  height: 80px;
}
</style>

<!-- export default {
  name: 'App',
  components: {
    HelloWorld,
    UploadFilled
  },
  data() {
    return {
      imgurl:{
        preview:"",
        plateview:""
      },
      plate:{
        res:"等待识别"
      },
      file:null
    }
  },
  methods:{
    delimg(){
      console.log(this.file)
      console.log(this.$refs.upload.value)
    },
    beforeUpload(){

    },
    handleExceed(files){
      this.$refs.upload.value.upload.value.clearFiles()
      const file = files[0]
      this.$refs.upload.value.upload.value.handleStart(file)
      this.file = file
    }
  }
} -->