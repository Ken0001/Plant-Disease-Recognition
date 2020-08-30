// File Upload
// 
function ekUpload() {
  function Init() {

    console.log("Upload Initialised");

    var fileSelect = document.getElementById('file-upload'),
      fileDrag = document.getElementById('file-drag'),
      submitButton = document.getElementById('submit-button');

    fileSelect.addEventListener('change', fileSelectHandler, false);

    // Is XHR2 available?
    var xhr = new XMLHttpRequest();
    if (xhr.upload) {
      // File Drop
     /* fileDrag.addEventListener('dragover', fileDragHover, false);
      fileDrag.addEventListener('dragleave', fileDragHover, false);
      fileDrag.addEventListener('drop', fileSelectHandler, false);*/
    }
  }

  function fileDragHover(e) {
    var fileDrag = document.getElementById('file-drag');

    e.stopPropagation();
    e.preventDefault();

    fileDrag.className = (e.type === 'dragover' ? 'hover' : 'modal-body file-upload');
  }

  function fileSelectHandler(e) {
    // Fetch FileList object
    var files = e.target.files || e.dataTransfer.files;

    // Cancel event and hover styling
    fileDragHover(e);

    // Process all File objects
    for (var i = 0, f; f = files[i]; i++) {
      parseFile(f);
      uploadFile(f);
    }
  }

  // Output
  function output(msg) {
    // Response
    var m = document.getElementById('messages');
    m.innerHTML = msg;
  }

  function parseFile(file) {

    console.log(file.name);
    output(
      '<strong>' + encodeURI(file.name) + '</strong>'
    );

    // var fileType = file.type;
    // console.log(fileType);
    var imageName = file.name;

    var isGood = (/\.(?=gif|jpg|png|jpeg)/gi).test(imageName);
    if (isGood) {
      document.getElementById('start').classList.add("hidden");
      document.getElementById('response').classList.remove("hidden");
      document.getElementById('notimage').classList.add("hidden");
      // Thumbnail Preview
      document.getElementById('file-image').classList.remove("hidden");
      document.getElementById('file-image').src = URL.createObjectURL(file);
    }
    else {
      document.getElementById('file-image').classList.add("hidden");
      document.getElementById('notimage').classList.remove("hidden");
      document.getElementById('start').classList.remove("hidden");
      document.getElementById('response').classList.add("hidden");
      document.getElementById("file-upload-form").reset();
    }
  }

  function setProgressMaxValue(e) {
    var pBar = document.getElementById('file-progress');

    if (e.lengthComputable) {
      pBar.max = e.total;
    }
  }

  function updateFileProgress(e) {
    var pBar = document.getElementById('file-progress');

    if (e.lengthComputable) {
      pBar.value = e.loaded;
    }
  }

  function uploadFile(file) {
   

    var xhr = new XMLHttpRequest(),
      fileInput = document.getElementById('class-roster-file'),
      pBar = document.getElementById('file-progress'),
      fileSizeLimit = 1024; // In MB
    if (xhr.upload) {
      // Check if file is less than x MB
      if (file.size <= fileSizeLimit * 1024 * 1024) {
        // Progress bar
        pBar.style.display = 'inline';
        xhr.upload.addEventListener('loadstart', setProgressMaxValue, false);
        xhr.upload.addEventListener('progress', updateFileProgress, false);

        // File received / failed
        xhr.onreadystatechange = function (e) {
          if (xhr.readyState == 4) {
            // Everything is good!

            // progress.className = (xhr.status == 200 ? "success" : "failure");
            // document.location.reload(true);
          }
        };

        var elem = document.getElementById('modal1');
        var instances = M.Modal.getInstance(elem);
        var form_data = new FormData($('#file-upload-form')[0]);

        
        console.log(form_data);
        console.log(file)
        $.ajax({
          xhr: function () {
            var xhr = new window.XMLHttpRequest();

            xhr.upload.addEventListener('progress', updateFileProgress, false);

            return xhr;
          },
          type: "POST",
          processData: false,
          url: "http://134.208.3.54:7788",
          data: form_data,
          contentType: false,
          cache: false,
          success: function (response) {
            console.log("success")
            //alert(response);
            $("#result").empty();
            $("#result").append(response);
            instances.open();
            
            console.log($("#file-image")[0].height)
            if ($("#file-image")[0].height > $("#file-image")[0].width) { 
              $("#file-image").addClass("long-picture");
              console.log("long")
            } else {
              $("#file-image").removeClass("long-picture");
            }
          },
          error: function (xhr) {
            console.log("failed")
            alert(xhr);
          }
        })

        

        // Start upload
        /*xhr.open('POST', "http://134.208.3.54:7788", true);
        xhr.setRequestHeader('X-File-Name', file.name);
        xhr.setRequestHeader('X-File-Size', file.size);
        xhr.setRequestHeader('Content-Type', 'multipart/form-data');
        xhr.send(file);*/

      } else {
        output('Please upload a smaller file (< ' + fileSizeLimit + ' MB).');
      }
    }
  }

  // Check for the various File API support.
  if (window.File && window.FileList && window.FileReader) {
    Init();
  } else {
    document.getElementById('file-drag').style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', function() {
  var elems = document.querySelectorAll('.modal');
  var options = {"endingTop": "10%"}
  var instances = M.Modal.init(elems, options);
});
ekUpload();

/*$(document).ready(function () {
  $('.modal').modal();
  $("#file-upload").on("change", function (event) {
    var form = $(this);
    var url = "http://134.208.3.54:7788";
    console.log(event)
    $.ajax({
      type: "POST",
      url: url,
      data: form.serialize(),
      success: function (response) {
        console.log("success")
        //alert(response);
        $("#modal1").append(response)
      },
      error: function () {
        console.log("failed")
        alert("!?");
      }
    });
  });
});*/
