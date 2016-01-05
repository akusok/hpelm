jQuery(document).ready(function($) {
	$('#button1').click(function () {	
		if ($('div').hasClass('open')) {
			if ($(this).hasClass('active')) {
				$('.menuBox').slideUp();
				$('.active').removeClass('active');
				$('.open').removeClass('open');
				$('.menuPage').fadeOut();
				$("body").css("background-position", "0 -45px");
			}
			else {
				$('img.dpArrowFour').hide();
				$('img.dpArrowFive').hide();
				$('img.dpArrowSix').hide();
				$('img.active').removeClass('active');
			$('#menuPage1').animate({left: '0px'}, 500 );
			$('#menuPage2').animate({left: '1092px'}, 500 );
			$('#menuPage3').animate({left: '2184px'}, 500 );
			$('#menuPage4').animate({left: '3276px'}, 500 );
			$('#menuPage5').animate({left: '4368px'}, 500 );
				$(this).addClass('active');
			}
		}
		else {
			$('img.dpArrowFour').hide();
			$('img.dpArrowFive').hide();
			$('img.dpArrowSix').hide();
			$('div.menuBox').slideDown();
			$('div.menuBox').addClass('open');
			$('div.menuPage').fadeIn(800);
			$('#menuPage1').css({left: '0px'});
			$('#menuPage2').css({left: '1092px'});
			$('#menuPage3').css({left: '2184px'});
			$('#menuPage4').css({left: '3276px'});
			$('#menuPage5').css({left: '4368px'});
			$(this).addClass('active');
			$("body").css("background-position", "0 -84px");
		}
	});
	
	$('#button2').click(function () {	
		if ($('div').hasClass('open')) {
			if ($(this).hasClass('active')) {
				$('.menuBox').slideUp();
				$('.active').removeClass('active');
				$('.open').removeClass('open');
				$('.menuPage').fadeOut();
				$("body").css("background-position", "0 -45px");
			}
			else {
				$('img.dpArrowFour').hide();
				$('img.dpArrowFive').hide();
				$('img.dpArrowSix').hide();
				$('img.active').removeClass('active');
				$('#menuPage1').animate({left: '-1092px'}, 500 );
			$('#menuPage2').animate({left: '0px'}, 500 );
			$('#menuPage3').animate({left: '1092px'}, 500 );
			$('#menuPage4').animate({left: '2184px'}, 500 );
			$('#menuPage5').animate({left: '3276px'}, 500 );
				$(this).addClass('active');
			}
		}
		else {
			$('img.dpArrowFour').hide();
			$('img.dpArrowFive').hide();
			$('img.dpArrowSix').hide();
			$('div.menuBox').slideDown();
			$('div.menuBox').addClass('open');
			$('div.menuPage').fadeIn(800);
			$('#menuPage1').css({left: '-1092px'});
			$('#menuPage2').css({left: '0px'});
			$('#menuPage3').css({left: '1092px'});
			$('#menuPage4').css({left: '2184px'});
			$('#menuPage5').css({left: '3276px'});
			$(this).addClass('active');
			$("body").css("background-position", "0 -84px");
		}
	});
	
	$('#button3').click(function () {	
		if ($('div').hasClass('open')) {
			if ($(this).hasClass('active')) {
				$('.menuBox').slideUp();
				$('.active').removeClass('active');
				$('.open').removeClass('open');
				$('.menuPage').fadeOut();
				$("body").css("background-position", "0 -45px");
			}
			else {
				$('img.dpArrowFour').hide();
				$('img.dpArrowFive').hide();
				$('img.dpArrowSix').hide();
				$('img.active').removeClass('active');
				$('#menuPage1').animate({left: '-2184px'}, 500 );
			$('#menuPage2').animate({left: '-1092px'}, 500 );
			$('#menuPage3').animate({left: '0px'}, 500 );
			$('#menuPage4').animate({left: '1092px'}, 500 );
			$('#menuPage5').animate({left: '2184px'}, 500 );
				$(this).addClass('active');
			}
		}
		else {
			$('img.dpArrowFour').hide();
			$('img.dpArrowFive').hide();
			$('img.dpArrowSix').hide();
			$('div.menuBox').slideDown();
			$('div.menuBox').addClass('open');
			$('div.menuPage').fadeIn(800);
			$('#menuPage1').css({left: '-2184px'});
			$('#menuPage2').css({left: '-1092px'});
			$('#menuPage3').css({left: '0px'});
			$('#menuPage4').css({left: '1092px'});
			$('#menuPage5').css({left: '2184px'});
			$(this).addClass('active');
			$("body").css("background-position", "0 -84px");
		}
	});
	
	$('#button4').click(function () {	
		if ($('div').hasClass('open')) {
			if ($(this).hasClass('active')) {
				$('.menuBox').slideUp();
				$('.active').removeClass('active');
				$('.open').removeClass('open');
				$('.menuPage').fadeOut();
				$("body").css("background-position", "0 -45px");
			}
			else {
				$('img.dpArrowFour').hide();
				$('img.dpArrowFive').hide();
				$('img.dpArrowSix').hide();
				$('img.active').removeClass('active');
				$('#menuPage1').animate({left: '-3276px'}, 500 );
			$('#menuPage2').animate({left: '-2184px'}, 500 );
			$('#menuPage3').animate({left: '-1092px'}, 500 );
			$('#menuPage4').animate({left: '0px'}, 500 );
			$('#menuPage5').animate({left: '1092px'}, 500 );
				$(this).addClass('active');
			}
		}
		else {
			$('img.dpArrowFour').hide();
			$('img.dpArrowFive').hide();
			$('img.dpArrowSix').hide();
			$('div.menuBox').slideDown();
			$('div.menuBox').addClass('open');
			$('div.menuPage').fadeIn(800);
			$('#menuPage1').css({left: '-3276px'});
			$('#menuPage2').css({left: '-2184px'});
			$('#menuPage3').css({left: '-1092px'});
			$('#menuPage4').css({left: '0px'});
			$('#menuPage5').css({left: '1092px'});
			$(this).addClass('active');
			$("body").css("background-position", "0 -84px");
		}
	});
	
	$('#button5').click(function () {	
		if ($('div').hasClass('open')) {
			if ($(this).hasClass('active')) {
				$('.menuBox').slideUp();
				$('.active').removeClass('active');
				$('.open').removeClass('open');
				$('.menuPage').fadeOut();
				$("body").css("background-position", "0 -45px");
			}
			else {
				$('img.dpArrowFour').hide();
				$('img.dpArrowFive').hide();
				$('img.dpArrowSix').hide();
				$('img.active').removeClass('active');
				$('#menuPage1').animate({left: '-4368px'}, 500 );
			$('#menuPage2').animate({left: '-3276px'}, 500 );
			$('#menuPage3').animate({left: '-2184px'}, 500 );
			$('#menuPage4').animate({left: '-1092px'}, 500 );
			$('#menuPage5').animate({left: '0px'}, 500 );
				$(this).addClass('active');
			}
		}
		else {
			$('img.dpArrowFour').hide();
			$('img.dpArrowFive').hide();
			$('img.dpArrowSix').hide();
			$('div.menuBox').slideDown();
			$('div.menuBox').addClass('open');
			$('div.menuPage').fadeIn(800);
			$('#menuPage1').css({left: '-4368px'});
			$('#menuPage2').css({left: '-3276px'});
			$('#menuPage3').css({left: '-2184px'});
			$('#menuPage4').css({left: '-1092px'});
			$('#menuPage5').css({left: '0px'});
			$(this).addClass('active');
			$("body").css("background-position", "0 -84px");
		}
	});
	
	$('img.close').click(function () {
		$('div.open').slideUp('slow');
		$('div.menuPage').fadeOut();
		$('div.open').removeClass('open');
		$('img.active').removeClass('active');
		$("body").css("background-position", "0 -45px");
	});
	
	$('a.h2').hover(function () {$(this).parent().css({'border': '2px solid #ff9200'});}, function () {$(this).parent().css({'border': '2px solid #000'});});
	$('a.h2').hover(function () {$(this).css({'color': '#ff9200'});}, function () {$(this).css({'color': '#000'});});
	$('a.h2').hover(function () {$(this).children().css({'color': '#ff9200'});}, function () {$(this).children().css({'color': '#000'});});
	
	$('a.dpLinks').hover(function () {
		$('img.dpArrowOne').fadeOut();
		$('img.dpArrowFour').fadeIn();
		$('img.dpArrowTwo').delay(100).fadeOut();
		$('img.dpArrowFive').delay(100).fadeIn();
		$('img.dpArrowThree').delay(200).fadeOut();
		$('img.dpArrowSix').delay(200).fadeIn();
	}, 
	function () {
		$('img.dpArrowFour').fadeOut();
		$('img.dpArrowOne').fadeIn();
		$('img.dpArrowFive').delay(100).fadeOut();
		$('img.dpArrowTwo').delay(100).fadeIn();
		$('img.dpArrowSix').delay(200).fadeOut();
		$('img.dpArrowThree').delay(200).fadeIn();
	});
	
	$('#button1').hover(function () {
        this.src = '/image/menu/products_button_on.jpg';
    }, function () {
        this.src = '/image/menu/products_button_off.jpg';
    });
	$('#button2').hover(function () {
        this.src = '/image/menu/support_button_on.jpg';
    }, function () {
        this.src = '/image/menu/support_button_off.jpg';
    });
	$('#button3').hover(function () {
        this.src = '/image/menu/community_button_on.jpg';
    }, function () {
        this.src = '/image/menu/community_button_off.jpg';
    });
	$('#button4').hover(function () {
        this.src = '/image/menu/MemHome_button_on.jpg';
    }, function () {
        this.src = '/image/menu/MemHome_button_off.jpg';
    });
	$('#closebutton').hover(function () {
        this.src = '/image/menu/close_on.png';
    }, function () {
        this.src = '/image/menu/close_off.png';
    });
});