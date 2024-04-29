
gsap.registerPlugin(ScrollTrigger);

const hamburger = document.querySelector(".hamburger");
const links = document.querySelector(".nav-links");

const handleClick = () => {
  hamburger.classList.toggle("toggle");
  links.classList.toggle("open");
};

hamburger.addEventListener("click", handleClick);


const lenis = new Lenis()

lenis.on('scroll', (e) => {
  console.log(e)
})

function raf(time) {
  lenis.raf(time)
  requestAnimationFrame(raf)
}

requestAnimationFrame(raf)


const swiper = new Swiper('.sample-slider', {
  effect: "cards",    //added
  grabCursor: true,   //added
  pagination: {
      el: '.swiper-pagination',
  },
  navigation: {
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev",
  },
  
})

function loaderAnimation() {
    var loader = document.querySelector("#loader")
    setTimeout(function () {
        loader.style.top = "-100%"
    }, 2500)
}

loaderAnimation()

// scroll animation

gsap.to(".hero>h1", {
  opacity: 1,
  delay: 2.5,
  y: 40,
  duration: 0.3,
  ease: "EaseIn",
});

gsap.to(".hero>h1", {
  backgroundPosition : "100%",
  scrollTrigger:{
    trigger:".hero",
    start:"center center",
    end: "bottom top",
    scrub:1,
    // markers:true,
    ease:"EaseIn",
 },
});

// gsap.to(".about::before", {
//   top : "-100%",
//   scrollTrigger:{
//     trigger:".about",
//     start:"center center",
//     end: "center center",
//     scrub:1,
//     markers:true,
//     ease:"EaseIn",
//  },
// });




// linking code

  swiper.on("slideChange", function () {
    var slideIndex = swiper.realIndex ;
    // console.log(slideIndex);
    var paragraphs = document.querySelectorAll(".features>.left>.paragraph");
    paragraphs.forEach(function (paragraph, index) {
      console.log(index);
      if (index === slideIndex) {
            paragraph.classList.add("active");
        } else {
            paragraph.classList.remove("active");
        }
    });
});
   