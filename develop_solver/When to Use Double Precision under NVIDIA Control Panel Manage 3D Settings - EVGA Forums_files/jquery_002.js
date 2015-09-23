/**
 * LavaLamp - A menu plugin for jQuery with cool hover effects.
 * @requires jQuery v1.1.3.1 or above
 *
 * http://gmarwaha.com/blog/?p=7
 *
 * Copyright (c) 2007 Ganeshji Marwaha (gmarwaha.com)
 * Dual licensed under the MIT and GPL licenses:
 * http://www.opensource.org/licenses/mit-license.php
 * http://www.gnu.org/licenses/gpl.html
 *
 * Version: 0.2.0
 * Requires Jquery 1.2.1 from version 0.2.0 onwards. 
 * For jquery 1.1.x, use version 0.1.0 of lavalamp
 */

/**
 * Creates a menu with an unordered list of menu-items. You can either use the CSS that comes with the plugin, or write your own styles 
 * to create a personalized effect
 *
 * The HTML markup used to build the menu can be as simple as...
 *
 *       <ul class="lavaLamp">
 *           <li><a href="#">Home</a></li>
 *           <li><a href="#">Plant a tree</a></li>
 *           <li><a href="#">Travel</a></li>
 *           <li><a href="#">Ride an elephant</a></li>
 *       </ul>
 *
 * Once you have included the style sheet that comes with the plugin, you will have to include 
 * a reference to jquery library, easing plugin(optional) and the LavaLamp(this) plugin.
 *
 * Use the following snippet to initialize the menu.
 *   $(function() { $(".lavaLamp").lavaLamp({ fx: "backout", speed: 700}) });
 *
 * Thats it. Now you should have a working lavalamp menu. 
 *
 * @param an options object - You can specify all the options shown below as an options object param.
 *
 * @option fx - default is "linear"
 * @example
 * $(".lavaLamp").lavaLamp({ fx: "backout" });
 * @desc Creates a menu with "backout" easing effect. You need to include the easing plugin for this to work.
 *
 * @option speed - default is 500 ms
 * @example
 * $(".lavaLamp").lavaLamp({ speed: 500 });
 * @desc Creates a menu with an animation speed of 500 ms.
 *
 * @option click - no defaults
 * @example
 * $(".lavaLamp").lavaLamp({ click: function(event, menuItem) { return false; } });
 * @desc You can supply a callback to be executed when the menu item is clicked. 
 * The event object and the menu-item that was clicked will be passed in as arguments.
 */
(function ($) {
    $.fn.lavaLamp = function (options) {

        var defaultOptions = { easing: "linear", speed: 500,
            selectedLiClass: 'selected',
            lavaSelected: 'lavaselected', lavaLiClass: 'lavaback', lavaDivClass: 'lavaleft',
            click: function () { }
        };

        var o = $.extend(true, {}, defaultOptions, options);

        return this.each(function () {
            var $ul = $(this),
            noop = function () { },
            $lavaBackLi = $('<li class="' + o.lavaLiClass + '"><div class="' + o.lavaDivClass + '"></div></li>').appendTo($ul),
            $lists = $("li", this),
            $currentLi = $lists.filter('.' + o.selectedLiClass); // || $($li[0]).addClass(o.selectedClass)[0];

            var zindex = [8, 10, 12];

            $lists.not("." + o.lavaLiClass).hover(function () {
                var $currli = $(this);
                setBackLiClass($currli);
                moveLavaTo($currli);
            }, noop).css('z-index', zindex[1]);

            $ul.hover(function () {
                if ($lists.filter('.' + o.selectedLiClass).length == 0) setBackLiUnselected();
            }, function () {
                moveLavaTo($currentLi);
            });

            $ul.delegate('li:not(.' + o.lavaLiClass + ')', 'click', function (e) {
                setCurrent($(this));
                setBackLiSelected();
                o.click(e, this);
            });

            $lavaBackLi.click(function () { return false; });

            setCurrent($currentLi);

            function setCurrent($el) {
                $lavaBackLi.css({ "left": ($el.length > 0 ? $el.position().left : -500) + "px", "width": $el.width() + "px" });
                if ($el.length == 0) {
                    $lavaBackLi.hide();
                    return;
                }
                $currentLi = $el;
            };

            function moveLavaTo($el) {
                $lavaBackLi.each(function () { $(this).dequeue(); });
                setBackLiClass($el);
                $lavaBackLi.show().animate({ width: $el.width(), left: ($el.length > 0 ? $el.position().left : -500) }, o.speed, o.easing);
                if ($el.length == 0) $lavaBackLi.hide();
            };

            function setBackLiClass($selectedLi) {
                if ($selectedLi.hasClass(o.selectedLiClass)) setBackLiSelected(); else setBackLiUnselected();
            }

            function setBackLiSelected() {
                $lavaBackLi.addClass(o.lavaSelected).css('z-index', zindex[2]);
            }

            function setBackLiUnselected() {
                $lavaBackLi.removeClass(o.lavaSelected).css('z-index', zindex[0]);
            }
        });
    };
})(jQuery);
