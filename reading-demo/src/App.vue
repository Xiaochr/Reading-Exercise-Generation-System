<template>
  <div id="app">
    <el-header>
      <el-text class="mx-1" size="large" tag="b">Reading Comprehension Exercise Generation</el-text>
    </el-header>
    <el-space :size="50">
    
    
    <el-card class="index-card">
      <el-header>
        Requirements
      </el-header>
      <el-form :inline="true">
        <el-form-item label="Topics: " label-width="80px">
          <el-input v-model="topics" placeholder=""></el-input>
        </el-form-item>
        <el-form-item label="CEFR: " label-width="80px">
          <el-radio-group v-model="cefr">
            <el-radio-button label="A1">A1</el-radio-button>
            <el-radio-button label="A2">A2</el-radio-button>
            <el-radio-button label="B1">B1</el-radio-button>
            <el-radio-button label="B2">B2</el-radio-button>
          </el-radio-group>
        </el-form-item>
      </el-form>

      <el-form :inline="true">
        <el-form-item label="Length: " label-width="80px">
          <el-input v-model="essay_len" placeholder=""></el-input>
        </el-form-item>
        <el-form-item label="Genre: " label-width="80px">
          <!-- <el-input v-model="genres" placeholder=""></el-input> -->
          <el-select v-model="genres">
            <el-option
              v-for="item in genre_options"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
      </el-form>

      <el-divider></el-divider>

      <el-form label-width="80px">
        <el-form-item label="Based on the example?" label-width="180px">
          <el-radio-group v-model="is_example">
            <el-radio-button label="1">Yes</el-radio-button>
            <el-radio-button label="0">No</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="Example: ">
          <el-input 
            v-model="example"
            :rows="8" 
            type="textarea"
            placeholder="">
          </el-input>
        </el-form-item>
      </el-form>

      <el-divider></el-divider>

      <el-form label-width="80px">
        <el-form-item label="# Questions" label-width="100px">
          <el-radio-group v-model="q_num">
            <el-radio-button label="1">1</el-radio-button>
            <el-radio-button label="2">2</el-radio-button>
            <el-radio-button label="3">3</el-radio-button>
            <el-radio-button label="4">4</el-radio-button>
            <el-radio-button label="5">5</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="# Choices" label-width="100px">
          <el-radio-group v-model="a_num">
            <el-radio-button label="3">3</el-radio-button>
            <el-radio-button label="4">4</el-radio-button>
            <el-radio-button label="5">5</el-radio-button>
          </el-radio-group>
        </el-form-item>
        
        <el-form-item label="Question Type: " label-width="120px">
          <el-select v-model="q_type">
            <el-option
              v-for="item in type_options"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        
      </el-form>

      
    </el-card>

    <el-card class="index-card">
      <el-header>Generated Exercise</el-header>
      <el-scrollbar height="650px">
      <el-form label-width="80px">
        <el-form-item label="Passage: ">
          <el-input 
            v-model="essay"
            :rows="11" 
            type="textarea"
            placeholder="">
          </el-input>
        </el-form-item>

        <el-button type="primary" primary @click="gen_essay">Generate</el-button>
        <el-button type="primary" plain @click="clear_essay">Clear</el-button>

        <el-divider></el-divider>

        <el-form-item label="Questions: ">
          <el-input 
            v-model="questions"
            :rows="11" 
            type="textarea"
            placeholder="">
          </el-input>
        </el-form-item>
      </el-form>
      <el-button type="primary" primary @click="gen_questions">Generate</el-button>
      <el-button type="primary" plain @click="clear_questions">Clear</el-button>
      </el-scrollbar>
      
    </el-card>
    </el-space>
  </div>
</template>

<script>
import qs from 'qs'
export default {
  data() {
    return {
      topics: 'sports, healthy',
      cefr: 'A2',
      essay_len: 200,
      genres: 'narrative',
      is_example: '0',
      example: '',
      essay: '', 
      questions: '',
      q_num: '4',
      a_num: '4',
      q_type: 'random',
      genre_options: [
        {value: "narrative", label: "narrative"},
        {value: "descriptive", label: "descriptive"},
        {value: "expository", label: "expository"},
        {value: "persuasive", label: "persuasive"},
        {value: "instructive", label: "instructive"},
        {value: "news", label: "news"},
        {value: "opinion", label: "opinion"},
        {value: "conversation", label: "conversation"},
        {value: "letter/email", label: "letter/email"},
        {value: "poster", label: "poster"},
      ],
      type_options: [
        {value: "random", label: "random"},
        {value: "vocabulary", label: "vocabulary"},
        {value: "inference", label: "inference"},
        {value: "purpose", label: "purpose"},
        {value: "emotion", label: "emotion"},
        {value: "structure", label: "structure"},
      ],
    }
  },
  methods: {
    gen_essay() {
      console.log("generating essay")
      this.$http.post('http://127.0.0.1:8000/app/gen_essay/', 
        qs.stringify(
            {
              "topics": this.topics,
              "cefr": this.cefr,
              "essay_len": this.essay_len,
              "genres": this.genres, 
              "is_example": this.is_example,
              "example": this.example,
            }
          )
        ).then(
        (rep) => {
          console.log(rep)
          this.essay = rep.data
        }
      )
    },
    gen_questions() {
      console.log("generating questions")
      this.$http.post('http://127.0.0.1:8000/app/gen_questions/', 
        qs.stringify(
            {
              "essay": this.essay,
              "q_num": this.q_num,
              "a_num": this.a_num,
              "q_type": this.q_type,
            }
          )
        ).then(
        (rep) => {
          console.log(rep)
          this.questions = rep.data
        }
      )
    },
    clear_essay() {
      this.essay = ""
    },
    clear_questions() {
      this.questions = ""
    },
    
    
    init_func() {
      console.log("")
    }
  },
  mounted() {
    this.init_func()
  }
}
</script>

<style lang="css">
.index-card {
  width: 600px;
  margin: 30px auto;
}
.el-row {
  margin-bottom: 10px;
}
.el-form-item {
  margin-bottom: 15px;
}
.el-radio-group {
  margin-bottom: 0px;
}
.el-radio {
  margin-bottom: 10px;
}
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 20px;
}
body {
  background: #F5F5F5;
}
</style>
