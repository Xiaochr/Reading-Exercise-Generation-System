import * as Vue from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'
import VueAxios from 'vue-axios'
import App from './App.vue'

const app = Vue.createApp(App)

// const cors = require("cors");
// app.use(cors());

app.use(ElementPlus)
app.use(VueAxios, axios)

app.mount('#app')
