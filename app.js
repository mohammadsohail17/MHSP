const express=require('express')
const bodyparser=require('body-parser')
const path=require('path')
const app=express()

app.use(bodyparser.urlencoded({'extended':true}))
const spawn = require("child_process").spawn;
//const pythonProcess = spawn('python',["--version"]);

const pythonProcess=spawn('python',[__dirname+'/model/mlmodel.ipynb'])
app.use(express.static(path.join(__dirname, 'public')))
pythonProcess.stdout.on('data',(data)=>{
    console.log(data.toString())
}
)
pythonProcess.stderr.on('data',(data)=>{
    console.error('stderr:',data.toString())
})
app.get("/",function(req,res){
    res.sendFile(__dirname+'/index.html')
})

app.get("/about",function(req,res){
    res.sendFile(__dirname+"/about.html")
})
app.get("/contact",function(req,res){
    res.send("<h2>Contact</h2>")
})

  
app.post("/post", (req, res) => {
    var name=req.body
    console.log(name)
    //res.redirect('/about')
  console.log("Connected to React");
  res.redirect("/");
});
  
const PORT = process.env.PORT || 9000;
  
app.listen(PORT, console.log(`Server started on port ${PORT}`));