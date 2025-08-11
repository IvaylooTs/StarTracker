const skipLogo = true

window.onload = () => {

  document.getElementById("display-cubeCheckBox").checked = true;
  document.getElementById("display-cubeDirectionVector").checked = true;
  document.getElementById("display-CordinateSystem").checked = true;
  document.getElementById("display-CalibrationVectors").checked = true;
  document.getElementById("display-newCalibrationVectors").checked = true;
  document.getElementById("display-oldCalibrationVectors").checked = true;
  if(skipLogo == true){
    document.getElementById('load-screen').style.display = 'none';
    return
  }
  
  element = document.getElementById("text-container");
  element.children[0].textContent = "Developer interface";

    
    if(typeof anime === 'undefined'){
        element = document.getElementById("logo");
        console.log("not found");
        element.style.opacity = 1;

        element = document.getElementById("text-container");
        element.style.opacity = 1;
        const delay = ms => new Promise(res => setTimeout(res, ms));
        const yourFunction = async () => {
            await delay(1000);
            element = document.getElementById("load-screen");
            element.style.opacity = 0;
            document.getElementById('load-screen').style.display = 'none';
        };
        yourFunction();
        
        return;
    }


    const timeline = anime.timeline({
      easing: 'easeInOutQuad',
      duration: 1200
    });

    // Move logo to left
    timeline.add({
      targets: '#logo',
      opacity: [0, 1],
    //   translateX: '-20%', // moves to the left
      duration: 500,
      offset:0
    }).add({
      targets: '#text-container',
      opacity: [0, 1],
      translateX: ['-2%', '2%'], // from right to center
      duration: 500,
      offset: 1 // overlap with logo slide
    }).add({
        targets: '#load-screen',
        opacity:[1,0],
        duration: 200,
        delay:1000,
        complete: function() {
            document.getElementById('load-screen').style.display = 'none';
            document.getElementById('stream').src = 'https://192.168.55.160:5000/video_feed';
        }
    });
  };
