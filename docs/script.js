function fcnResize() {
    let bodyWidth = document.body.offsetWidth.valueOf();
    main.style.marginLeft = bodyWidth / 20 + 'px';
    
    console.log(bodyWidth);
}