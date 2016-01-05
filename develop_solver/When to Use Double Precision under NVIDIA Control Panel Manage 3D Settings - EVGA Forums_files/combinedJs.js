/*tm_key*/


/* http://www.kryogenix.org/code/browser/searchhi/ */
/* Modified 20021006 to fix query string parsing and add case insensitivity */
/* Modified 20070316 to stop highlighting inside nosearchhi nodes */

var ASPPG_searchQuery = 'high';

function highlightWord(node,word) {
    if (word.trim().length == 0) return;
	// Iterate into this nodes childNodes
	if (node.hasChildNodes) {
		var hi_cn;
		for (hi_cn=0;hi_cn<node.childNodes.length;hi_cn++) {
			highlightWord(node.childNodes[hi_cn],word);
		}
	}
	
	// And do this node itself
	if (node.nodeType == 3) { // text node
		tempNodeVal = node.nodeValue.toLowerCase();
		tempWordVal = new RegExp('\\b' + word.toLowerCase() + '\\b', "g");
		if (tempNodeVal.search(tempWordVal) != -1) {
			pn = node.parentNode;
			// check if we're inside a "nosearchhi" zone
			checkn = pn;
			while (checkn.nodeType != 9 && 
			checkn.nodeName.toLowerCase() != 'body') { 
			// 9 = top of doc
				if (checkn.className.match(/\bnosearchhi\b/)) { return; }
				checkn = checkn.parentNode;
			}
			if (pn.className != ASPPG_searchQuery) {
				// word has not already been highlighted!
				nv = node.nodeValue;
				ni = tempNodeVal.search(tempWordVal);
				// Create a load of replacement nodes
				before = document.createTextNode(nv.substr(0,ni));
				docWordVal = nv.substr(ni,word.length);
				after = document.createTextNode(nv.substr(ni+word.length));
				hiwordtext = document.createTextNode(docWordVal);
				hiword = document.createElement("span");
				hiword.className = ASPPG_searchQuery;
				hiword.appendChild(hiwordtext);
				pn.insertBefore(before,node);
				pn.insertBefore(hiword,node);
				pn.insertBefore(after,node);
				pn.removeChild(node);
			}
		}
	}
}

function searchHighlight() {
    var qs = getQueryString(ASPPG_searchQuery);
    var highlitewords = (qs) ? qs.trim() : '';

    highlitewords = highlitewords.replace(document.location.hash, '');

    if (typeof highlitewords != 'undefined' && highlitewords != '') {
        var wordList = highlitewords.replace(/\s/gi, ',').split(',');
        var nodes = $(".msg, .excerpt");

        for (var j = 0; j < nodes.length; j++) {
            highlightWord(nodes[j], highlitewords);

            for (var i = 0; i < wordList.length; i++) {
                highlightWord(nodes[j], wordList[i]);
            }
        }
    }  
}

$doc.ready(function () { searchHighlight(); });


    function RegisterMsgSideHover() {

        if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;

        $body.on('mouseenter', 'div.msgcontent', function () {

            var winHeight = $win.height(), distanceToShowFactor = 1.5;
            var $msgside = $(this);

            if ($msgside.height() < (winHeight * distanceToShowFactor)) return false;

            var $btnsPanel = $msgside.find('.msgButtonsPanel');
            var fixedClass = 'halfhidden';

            if ($btnsPanel.hasClass(fixedClass)) return false;

            var btnOffset = $btnsPanel.offset();

            if ($win.scrollTop() + (winHeight * distanceToShowFactor) <= btnOffset.top) {

                var $fixPanel = $btnsPanel.clone();

                $fixPanel.addClass(fixedClass).css({ left: btnOffset.left - 5 }).appendTo($msgside);

                $win.bind('scrollstop.btnpanel', function () {

                    var winScrollTop = $win.scrollTop();

                    if (winScrollTop + winHeight > btnOffset.top || winScrollTop + winHeight < $msgside.offset().top) {
                        $fixPanel.remove();
                        $win.unbind('scrollstop.btnpanel');
                    }
                });
            }

            return true;
        });
    }

    function RegisterMemberHover() {

        if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;

        var timeout;

        $body
            .on('mouseenter', '.authorcontent a.titlehead', function () {

                var $titleHeadLink = $(this);

                timeout = setTimeout(function () {

                    var $parentDiv = $titleHeadLink.closest('div.authorcontent');

                    var $overallDiv = $('<div />').addClass('titleheadHoverDiv');

                    var $div = $('<div />').addClass('titleheadOriginalDiv');

                    $div.html($parentDiv.html());
                    $div.find('.msgAuthorInfo').remove(); // full mode
                    $div.find('.msgStatus').remove(); // essential mode

                    $overallDiv.html($div);

                    var offset = $parentDiv.offset();

                    $overallDiv.css({
                        'top': (offset.top - 10) + 'px',
                        'left': offset.left + 'px'
                    });

                    $div.css({
                        'width': $parentDiv.width(),
                        'height': $parentDiv.height() + 15 - $parentDiv.find('.msgAuthorInfo').height(), // full mode
                        'max-width': 250
                    });

                    CreateMemberMenu($titleHeadLink, $overallDiv);

                    $overallDiv.appendTo('body');

                }, 500);

            })
            .on('mouseleave', '.authorcontent a.titlehead', function () {
                clearTimeout(timeout);
            })
            .on('mouseleave', 'div.titleheadHoverDiv', function () {
                var $container = $(this);
                setTimeout(function () {
                    $container.detach();
                }, 200);
            });
    
    }

    var _attachmentImageUrls = [];
    var _clickedThumbLink;
    function RegisterToolTipAttachmentImages() {

        if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;
        
        $body.on('click', '.attachments a.attachedImages', function (e) {
            e.preventDefault();

            _clickedThumbLink = this;
            var $this = $(this);

            getImgsIntoArray($this.parent().find('a.attachedImages'));
            toolTipWin(produceImgSrc($this));

        }).on('click', 'img.toolTipWin', function (e) {
            e.preventDefault();
            var $thisImg = $(this);
            var nextI = findAttachmentIndex($thisImg.attr('src'))[1];
            $thisImg.attr('src', _attachmentImageUrls[nextI].src);
            $thisImg.nextAll('.filename').text(_attachmentImageUrls[nextI].filename);
            setTimeout(function () { $thisImg.closest('div.qtip').qtip('api').reposition(); }, 100);

        }).on('click', 'a.toolTipExtlink', function () {
            var $this = $(this);
            $this.attr('href', $this.prev().attr('src'));

        }).on('click', 'a.toolTipRefreshlink', function () {
            var $thisImg = $(this).prevAll('img.toolTipWin');
            var src = $thisImg.attr('src');
            var random = randomString(5);
            var reloadSrc = src + '&refresh=' + random;

            $thisImg.attr('src', reloadSrc);

            getImgsIntoArray($(_clickedThumbLink).parent().find('.attachedImages'), random);

        });

        
        
        $body.on('mousewheel', 'img.toolTipWin', function(e, delta, deltaX, deltaY) {

            e.preventDefault();
            var $thisImg = $(this);
            //consoleLog(delta, deltaX, deltaY);
            var nextI = findAttachmentIndex($thisImg.attr('src'))[(deltaY < 0 ? 1 : 2)];
            //consoleLog(nextI);
            if (_attachmentImageUrls[nextI]) {
                $thisImg.attr('src', _attachmentImageUrls[nextI].src);
                $thisImg.nextAll('.filename').text(_attachmentImageUrls[nextI].filename);

                setTimeout(function() { $thisImg.closest('div.qtip').qtip('api').reposition(); }, 100);
            }

        });
    }

    function getImgsIntoArray($links, random) {
        _attachmentImageUrls = [];

        $links.each(function () {
            var $thisLink = $(this);
            var imgObj = {};
            imgObj.filename = $thisLink.data('filename');
            imgObj.src = produceImgSrc($thisLink) + (random ? '&refresh=' + random : '');
            _attachmentImageUrls[_attachmentImageUrls.length] = imgObj;
        });
    }

    function produceImgSrc($link) {
        return $link.attr('href') + '&filename=' + $link.data('filename');
    }

    function toolTipWin(url) {
        var winSizes = getViewportSize();
        var maxHeight = winSizes[1] - 80;
        qtip.pop(url, null, false, true, 7000,
            '<img src="{0}" class="block xMarginAuto toolTipWin" style="max-width: 800px; max-height: ' + maxHeight.toString() + 'px;" />' +
            '<a target="_blank" class="ui-state-default ui-corner-all right tMargin10 lMargin5 pointer toolTipExtlink"><span class="ui-icon ui-icon-extlink"></span></a> ' +
            '<a target="_blank" class="ui-state-default ui-corner-all right tMargin10 pointer toolTipRefreshlink"><span class="ui-icon ui-icon-refresh"></span></a> ' +
            '<div class="center tMargin10 bold filename"></div>',
                {
                    style: { classes: 'qtip-rounded qtip-wideshadow qtip-youtube' },
                    position: { effect: false },
                    hide: {
                        event: 'unfocus'
                    },
                    events: { 
		                render: function(event, api) {
			                $win.bind('keydown', function(e) {
   				                if(e.keyCode === 27) { api.hide(e); }
			                });
		                },                        
                        visible: function (e, api) {
                            var $this = $(this);
                            //$this.draggable();
                            api.reposition();
                            var $thisImg = $this.find('img.toolTipWin');
                            var thisI = findAttachmentIndex($thisImg.attr('src'))[0];
                            
                            if (_attachmentImageUrls[thisI])
                                $thisImg.nextAll('div.filename').text(_attachmentImageUrls[thisI].filename);
                        },
                        hide: function (e, api) {
                            api.destroy();
                        }
                    }
                });
    }

    function findAttachmentIndex(src) {
        var nextI = 0;
        var prevI = _attachmentImageUrls.length - 1;
        var thisI = 0;
        for (var i = 0; i < _attachmentImageUrls.length; i++) {
            if (_attachmentImageUrls[i].src == src) {
                thisI = i;
                if (i != (_attachmentImageUrls.length - 1)) {
                    nextI = i + 1;
                }
                if (i != 0) {
                    prevI = i - 1;
                }
                break;
            }
        }

        return [thisI, nextI, prevI];
    }
   
    function RefreshClick(val, id) { // val: reply / edit; id = msgID

        var setXmlHttpVal = '';
        
        switch (val) {

            case 'reply':
                // compare CurrentPageRecordCount value, if exceeding mpg, redirect
                if (parseInt($get('CurrentPageRecordCount').value) == tm.mpg) {
                    self.location.href = cPathInfo.ForumDir + 'FindPost/' + id.toString();
                    return false;
                } else {
                    setEditorContent('');
                }
        case 'edit':
                // simply refresh
            
            default:
            
        }

        tmRefreshMessageList((tm.inTreeMode ? $get(tm.msgTreeHiddenField).value : setXmlHttpVal));
        
        if(typeof id != 'undefined') {
            _scrollToMsgId = id;
        }
        
        return false;
    }

    function tmRefreshMessageList(val) {
        $('#ResultXmlPanel').css('opacity', 0.4);
        $find('ResultXmlPanel').set_value(val);
    }
    
    var _scrollToMsgId = 0;

    function afterRefresh() { // called from tm.aspx.vb; also executed on initial load

        var $itemToScrollTo = $('#msg' + _scrollToMsgId.toString());
        if ($itemToScrollTo.length > 0) $.scrollTo($itemToScrollTo);
        _scrollToMsgId = 0;
        
        // update pageLastGenerated - this and in tm.js is the only place where we make such change, 
        // for marking forum / thread read we cannot rely on server's now()
        pageLastGenerated = new Date().addHours(-_userTimeOffset);

        trackMessageRead($('.msgcontent span.msgDate').filter(':not(.none)').children('span.performdateformat'),
            function () { var mid = $(this).parent().data('mid');
                return $('#Avatar' + mid.toString()); });
    }

    function changeTmSort(value){
        if (isNaN(value)) return;
        SaveSortCookie(value, function (){tm.refreshButton.click();});
    }


    function reloadParent(arg) {
        
        if(typeof(arg) == 'string'){
            if (arg == '') {
                return;
            }
            else if (arg == 'true') {
                tm.refreshButton.click();
            }
            else if (arg == 'del') {
                RefreshAfterDelete();
            }
            else if (arg == 'reload') {
                self.location.replace(cPathInfo.Url);
            }
            else if (!isNaN(arg)) {
                self.location.replace(cPathInfo.ForumDir + 'tt.aspx?forumID=' + arg);
            }
            
        }

    }

    function RefreshAfterDelete() {
        $get(tm.msgLastMessageForPostHiddenField).value = '0'; // to force rebinding of the posting interface;
        
        if (!tm.inTreeMode) {
            tm.refreshButton.click();
        } else {
            treeProcessAfterDelete();
        }
    }

    function qtipPopupCloseHandler() {
        reloadParent($(this).data('args'));
    }

    function switchMsgDivOnOff(msgID, link) { // called in ItemTemplate.ascx
        $(link).parent().fadeOut('fast', function () {

            var $link = $(this);

            if (tm.inEssentialMode) {
                $link.closest('.msgtable').find('.msgcontent div').filter('.none').fadeIn('fast');
            } else {
                $link.siblings().fadeIn('fast');
            }
        });

        return false;
    }

	var multiQuoteIDs = [];

	function AddMultiQuote(msgID) {
	    
	    if (isNaN(msgID)) return;
	    
	    for (var i =0; i<multiQuoteIDs.length; i++){
	        if(multiQuoteIDs[i] == msgID) {return;}
        }	
	
	    if (multiQuoteIDs.length<3) {
	        multiQuoteIDs[multiQuoteIDs.length] = msgID;
	        $(".multiquoteImg" + msgID).hide().attr('src', pageThemeImageURL + 'menuUnMultiQuote.gif').show();
        } else {
            alert(tooManyQuotes);
        }
        
	}
	
	function RemoveMultiQuote(msgID){
	    if (isNaN(msgID)) return;
	    
	    var indexToDelete = -1;
	    
	    for (var i =0; i<multiQuoteIDs.length; i++){
	        if(multiQuoteIDs[i] == msgID) {indexToDelete = i; break;}
        }
        
        if (indexToDelete!=-1) multiQuoteIDs.splice(indexToDelete,1);
        $(".multiquoteImg" + msgID).fadeOut('slow');        
	}

    function replySimulatedClick(msgID, canReply){

        if (!canReply) {
            cMemberInfo.popupPosting = false;
        }

        var gotoURL = String.format('post.aspx?mq={0}&messageID={1}', multiQuoteIDs.join(), msgID);
        executePostReplyEditLink(gotoURL);

        return false;
    }

    function editSimulatedClick(msgID) {

        var gotoURL = String.format('post.aspx?edit=true&messageID={0}', msgID);
        executePostReplyEditLink(gotoURL);
        return false;
    }

    function executePostReplyEditLink(gotoURL) {

        if (cMemberInfo.popupPosting) {
            popRadWin(gotoURL + '&pop=true');
        } else {
            self.location.href = cPathInfo.ForumDir + gotoURL;
        }

        return false;
    }

    function showMsgNum(msgID) {
        popTip(msgID, 400, false, true, 5000,
        "<div><input type='text' onfocus='this.select()' style='width:350px;' value='" + 
            cPathInfo.ForumDir + "FindPost/{0}' /></div>", 'URL');
    }

    function RegisterNextPrevLinkInHeader() {//called directly in tm.aspx

        var html = String.format(
            '<div class="lMargin10 left">{4} <a href="{3}{0}?go=prev" data-dir="prev">{1}</a></div>' +
            '<div class="rMargin10 right"><a href="{3}{0}?go=next" data-dir="next">{2}</a> {5}</div>' +
            '<div class="divider"></div>',
            currentThreadID, ln.tmPrevThread, ln.tmNextThread, cPathInfo.ForumDir + 'FindPost/',
            ln.entUpArrow, ln.entDownArrow);

        var $nxtprv = $('div.nxtprv');


        if (!(cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice)) {
            var timeout;

            $nxtprv
                .on('mouseenter', 'a', function () {
                    var $link = $(this);
                    if ($link.data('retrieved')) return;
                    var linkDir = $link.data('dir');
                    timeout = setTimeout(function () {
                        $link.fadeOut('fast', function () {
                            JQCallWebService('ws/Message.aspx/GetNextPrev',
                                { msgID: currentThreadID, dir: linkDir },
                                function (r) {
                                    var returnedVal = r.d;

                                    if (returnedVal.text != '') {
                                        $link.text(returnedVal.text);
                                        $link.attr('href', returnedVal.url);
                                    } else {
                                        $link.attr('href', '#');
                                        $link.click(function () { return false; });
                                        qtip.notice($link, ln.NoDataWarning,
                                        {
                                            position: {
                                                my: 'middle ' + (linkDir == 'prev' ? 'left' : 'right'),
                                                at: 'middle ' + (linkDir == 'prev' ? 'right' : 'left')
                                            }
                                        });
                                    }

                                    $link.data('retrieved', true);
                                    $link.fadeIn();

                                }, JQOnCallError
                            );
                        });

                    }, 100);
                }).on('mouseleave', '.nxtprv a', function () {
                    clearTimeout(timeout);
                });            
        }

        $nxtprv.html(html);
    }

    function RegisterShowMarkTrack() {
        
        if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;
        
        var additionalOptions = {
            style: { classes: 'qtip-apgsmallmenu qtip-shadow' },
            hide: { delay: 600, event: 'unfocus mouseleave' },
            position: { my: 'top right', at: 'bottom right', adjust: { x: -10} },
            events: {
                render: function (e, api) {
                    var $tip = $(this);
                    $tip.find("a").click(function () { api.hide(); });
                }
            }
        };
        
        var func = function () {
            var $mark = $(this);
            var $markTrack = $mark.next('span');
            if ($markTrack.hasClass('msgMarkTrack')) {
                qtip.notice($mark,
                    $markTrack.html(), additionalOptions);

            }
        };

        $body.on('mouseenter', '.tmanswered, .tmhelpful', func);
    }

    $doc.ready(function () {
        if (typeof wysiwygAsTextarea != 'undefined')
            wysiwygAsTextarea.TextareaSelector = 'div#postdiv textarea:first';

        if (cMemberInfo.usingMobileTheme)
            $('#postdiv').css({ 'width': '95%', 'margin': '15px auto' }).find('textarea').css('width', '100%');
    });


    $doc.ready(function () {
        $(tm.refreshButton).click(RefreshClick);
        RegisterMsgSideHover();
        RegisterMemberHover();
        RegisterToolTipAttachmentImages();
        RegisterShowMarkTrack();
        RegisterMouseUpTextSelect();
        performOnPageGetScrollTo();
    });

    function performOnPageGetScrollTo() {
        var $initialScrollTo = $('#msg' + document.location.hash.replace('#', ''));

        if ($initialScrollTo[0]) {
            
            $initialScrollTo
                .effect('highlight', {}, 3000)
                .find('div.item, div.altItem')
                .effect('highlight', {}, 3000);

            $.scrollTo($initialScrollTo);
        }        
    }

    var tmSelectString;

    function RegisterMouseUpTextSelect() {

        if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;
        
        $body.bind('mouseup', function() {
            if (window.getSelection) {
                tmSelectString = window.getSelection().toString();
            }
            else if (window.document.selection) {
                var rng = window.document.selection.createRange();
                tmSelectString = rng.text;
            }
            else {
                tmSelectString = "";
            }
        });

    }

var _xmlPanelSel = '#ResultXmlPanel';
var _oriSearhcIDDataKey = 'orisearchid';
var _orisearchtermDataKey = 'orisearchterm';
var _relatedTopicULSel = '#relatedTopicsList';
var _relatedlnkSel = '#relatedSearch';
var _relatedResultDivSel = '#relatedTopics';
var _relatedRecordCount = 10;

function registerRelatedSearch(on, andExecute) { // registered in footertemplate
    var eToHandle = 'click.relatedSearch';

    if (on) {
        
        $(_xmlPanelSel).off(eToHandle, _relatedlnkSel).on(eToHandle, _relatedlnkSel, function () {

            var $lnk = $(this);
            var orisearchid = $lnk.data(_oriSearhcIDDataKey);

            if (orisearchid != 0) {
                getRelatedTopicsBySearchID(orisearchid, $lnk, true);
            } else {

                var termInLinkData = $lnk.data(_orisearchtermDataKey);

                var finalSearchString = termInLinkData != '' ? termInLinkData : ProduceSearchPhraseFromString(document.title, 4);

                initiateRelatedTopicSearch(finalSearchString, currentForumID, function (r) {
                    var result = r.d; //[asyncID, server now, highlight] or [Integer delay, "" ,""]
                    var aid = result[0];

                    if (aid == 'reentersearch') {
                        performNoRelatedTopicAction($lnk);
                        return false;
                    } else {
                        showNoticeToFilterSearchbox($lnk, ln.tmRelatedTopicsWaiting);
                        trackRelatedByAid(aid, result[1], $lnk);
                    }

                    return true;
                }, _relatedRecordCount);
            }

            return false;
        });

        //if (andExecute) setTimeout(function() { $(_relatedlnkSel).click(); }, 0);

        if (andExecute) {

            setTimeout(function () {
                var $link = $(_relatedlnkSel);

                if (!$link[0]) return;

                var relatedLinkOffsetTop = $link.offset().top;
                var relatedLinkHeight = $link.height();
                $win.bind('scrollstop.relatedTopics', function () {

                    var winScrollTop = $win.scrollTop();
                    var visibleLimit = winScrollTop + $win.height();

                    if (visibleLimit > relatedLinkOffsetTop && winScrollTop < (relatedLinkOffsetTop + relatedLinkHeight)) {
                        $link.data('notip', true);
                        $link.click();
                        $win.unbind('scrollstop.relatedTopics');
                    }

                });
            }, 3000);
        }

    } else {

        $(_xmlPanelSel).off(eToHandle, _relatedlnkSel).on(eToHandle, _relatedlnkSel, function () {
            var $ulRelatedList = $(_relatedTopicULSel);
            var $lnk = $(this);

            if ($ulRelatedList.length == 0) {
                //consoleLog('from off');
                displayNoRecordNotice($lnk);
            }
            else {
                //$.scrollTo($lnk);
            }

            return false;
        });
    }
}

function trackRelatedByAid(aid, beginTime, $lnk) {
    var relatedIntval = setInterval(function () {
        searchCheckIfComplete(aid, beginTime, function (result) {
            switch (result.d) {
                case 0: // keep checking
                    break;
                case -1:
                    clearInterval(relatedIntval);
                    performNoRelatedTopicAction($lnk);

                    break;
                default:
                    clearInterval(relatedIntval);
                    $lnk.qtip('hide');
                    storeRelatedSearchResults(result.d, function () {
                        getRelatedTopicsBySearchID(result.d, $lnk, false);
                    });
                    break;
            }
        });
    }, 1000);
}

var _relatedSearchRedoCount = 0;

function getRelatedTopicsBySearchID(searchID, $lnk, fromStorage) {

    //consoleLog(String.format('searchID: {0}, fromStorage: {1}', searchID, fromStorage));
    
    var ws = 'Message.aspx/RetrieveRelatedTopics';
    var ajaxData = { threadID: currentThreadID, top: _relatedRecordCount };

    JQCallWebService('ws/' + ws, ajaxData, function (r) {
        var sResults = r.d;

        var lis = '';
        for (var s = 0; s < sResults.length; s++) {

            var sResult = sResults[s];

            if (currentThreadID == sResult.MessageID)
                continue;

            lis += String.format('<li><div class="{3}"><a href="{0}Findpost/{1}">{2}</a></div></li>',
                cPathInfo.ForumDir, sResult.MessageID, sResult.Subject,
                (Modernizr.borderimage ? '' : '')); //Modernizr.borderimage as a test for IE all versions
        }

        if (lis == '') { // no data

            if (fromStorage && sResults.length == 0) { // re-search from subject

                if (_relatedSearchRedoCount == 0) {
                    $lnk.data(_oriSearhcIDDataKey, 0); // removed the data so that we can re-search
                    $lnk.click();
                    _relatedSearchRedoCount += 1;
                }

            } else {
                performNoRelatedTopicAction($lnk);
            }

        } else {

            var $relatedResultDiv = $(_relatedResultDivSel);
            
            $relatedResultDiv
                .find('ul')
                .replaceWith('<ul id="' + _relatedTopicULSel.replace('#', '') + '">' + lis + '</ul>')
                .end().fadeIn();

            $lnk.toggleClass('relatedTopics', true);
            
            registerRelatedSearch(false);
        }

    }, JQOnCallError);
}

function performNoRelatedTopicAction($lnk) {
    registerRelatedSearch(false);
    displayNoRecordNotice($lnk);
}

function displayNoRecordNotice($lnk) {
    
    var $relatedDiv = $(_relatedResultDivSel);
    
    $relatedDiv.show()
        .find('ul')
        .attr('id', _relatedTopicULSel.replace('#', ''))
        .html('<li>' + ln.tmRelatedTopicsNotFound + '</li>');

    $lnk.toggleClass('relatedTopics', true);
}

function storeRelatedSearchResults(searchID, callback) { // searchID = 0 =deleteonly
    //this performs 2 functions: 1) cache search results 2) if search results are fewer than 15, perform tag search
    JQCallWebService('ws/Message.aspx/StoreRelatedSearch',
                { msgID: currentThreadID, searchID: searchID, top: _relatedRecordCount },
                    ($.isFunction(callback) ? callback : null), JQOnCallError);
}

function registerRelatedCustomization() {
    var $lnk = $(_relatedlnkSel);
    var $tip;

    $(_xmlPanelSel).on('click', '#relatedCustomization', function () {

        var orisearchterm = $lnk.data(_orisearchtermDataKey);

        $tip =
            qtip.prompt($(this), 450, ln.tmRelatedTopicsCustomize, 
                (orisearchterm != '' ? orisearchterm : ProduceSearchPhraseFromString(document.title, 4)),
                ln.buttonUpdateValue, '', function (val) {
                    registerRelatedSearch(true);

                    $lnk
                        .data('notip', false)
                        .data(_oriSearhcIDDataKey, 0)
                        .data(_orisearchtermDataKey, val)
                        .click();

                    $tip.qtip('hide');                    
                }, null, {position:{at: 'top right'}});

        return false;
    });
    
}

$doc.ready(function () {
    registerRelatedCustomization();
});

function essentialModeToggle(on) {
    // Save to database by calling WS
    JQCallWebService(
        'ws/MessageList.aspx/SaveReadingPreference',
        { essentialsOnly: on, isThread: false },
        function () { tm.refreshButton.click(); },
        JQOnCallError);

    return false;
}

function quickQuote(msgID) {

    var $postDiv = $('#postdiv');
    
    if ($postDiv.length==0) {
        self.location.href = cPathInfo.ForumDir + 'post.aspx?quote=true&messageID=' + msgID;
        return false;
    }

    $get(tm.msgLastMessageForPostHiddenField).value = msgID;

    JQCallWebService("ws/Message.aspx/QuickQuote",
        { msgID: msgID },
        function (r) {
            if (typeof tinyMCE != 'undefined') { //WYSIWYG
                tinyMCE.activeEditor.insertHtml(r.d.WYSIWYG);
                tinyMCE.activeEditor.getBody().scrollTop = 0;

            } else { // textbox
                $(wysiwygAsTextarea.TextareaSelector).val(r.d.Text);
            }
            $.scrollTo($postDiv);
        });

    return false;    
}

function quickReplyAjax() {

    var draftId = parseInt($get(autoSave.ClientID).value) || 0;

    var $captchaControl = $('input[name$="captcha$txt"], input[id=recaptcha_response_field]').first();
    var captcha = $captchaControl.val() || '';

    var $reCaptchaChallenge = $('input[id=recaptcha_challenge_field]');
    var challenge = $reCaptchaChallenge.val() || '';

    JQCallWebService("ws/Message.aspx/QuickReply",
        { toMessageId: parseInt($get(tm.msgLastMessageForPostHiddenField).value),
            draftId: draftId,
            body: getEditorContent(), 
            attachmentID : '',
            captchaVal: captcha,
            challenge: challenge
        },
        function (r) {
            var returnedVal = r.d;
            switch (returnedVal.returnCode) {

                case 0:

                    KillSolutionOnlyCookie();
                    $('#previewdiv').hide();
                    window.clearInterval(_previewInterval); // defined in preview.js
                    
                    if (returnedVal.requiresApproval) {
                        sendFailureAlert(ln.postResultPendingApprovalDesc, 1500);
                        setTimeout(function() { self.location.replace(self.location.href); }, 3000);
                    } else {
                        setEditorContent('');

                        if (tm.inTreeMode) {
                            treeReload(returnedVal.newMsgID);
                        } else {
                            RefreshClick('reply', returnedVal.newMsgID);
                        }

                    }
                    break;

                default:
                    ///condensed all error messages here
                    var alerts = ['',
                        ln.postResultFailedFloodDesc,
                        ln.postResultFailedTopicLockedDesc,
                        ln.RightViolationMessage,
                        ln.formVerificationFailureMsg,
                        ln.warnRequiredDesc,
                        ln.postResultFailedExceedsPerDay];

                    sendFailureAlert(alerts[Math.abs(returnedVal.returnCode)]);
                    break;
            }

            if ($captchaControl.length == 1) {
                $captchaControl.next('img').click();
                if (typeof Recaptcha != 'undefined') Recaptcha.reload();
                if (returnedVal.returnCode != 0) $captchaControl.focus();
            }
        }
    );
}

function sendFailureAlert(txt, ms) {
    popTip(txt, null, true, false, ms || 4000);
}

function SaveSortCookie(sortDir, action) {
    JQCallWebService('ws/Message.aspx/SaveSortCookie', { sortDir: sortDir }, function () {
        if ($.isFunction(action)) action.call();
    }, JQOnCallError);
}

function SaveSolutionOnlyCookie(action) {
    JQCallWebService('ws/Message.aspx/SaveSolutionPreferenceCookie', { msgID: currentThreadID }, function () {
        if ($.isFunction(action)) action.call();
    }, JQOnCallError);
}

function KillSolutionOnlyCookie(action) {
    JQCallWebService('ws/Message.aspx/DeleteSolutionOnlyCookie', {}, function () {
        if ($.isFunction(action)) action.call();
    }, JQOnCallError);
}

var JQMenuLink;
var OptionalMenu;

function threadReadToggle(linkSelf, read) {
    $(linkSelf).css('cursor', 'progress');

    var data = { threadID: currentThreadID };

    if (read) data.mark = pageLastGenerated.format('s');

    JQCallWebService("ws/TrackRead.aspx/" + (read ? 'MarkThreadRead' : 'MarkThreadUnRead'), data,
        function (r) {
            setTimeout(function () { $(linkSelf).css('cursor', 'pointer'); }, 200);
            if (!r.d && !read) { qtip.alert(ln.tmOptionMarkUnReadUnAvailable); }
        }

    );

    return false;
}

function markApproval(msg, linkSelf) {

    $(linkSelf).css('cursor', 'progress');

    JQCallWebService("ws/Management.aspx/ApproveMessage", { msgIDs: [msg] },
    function(rtn) {
        var returnedResult = rtn.d;
        if (returnedResult === true) {
            if (msg == currentThreadID) {
                self.location.replace(self.location.href.replace(location.hash, ''));
            } else {
                tm.refreshButton.click();
            }
        }
    }, JQOnCallError);

}

function tmRestoreMessage(msgID) {
    var d = {};
    d.msgIDs = [msgID];
    tmCallManagementWS('RestoreMessage', d);
    return false;
}

function tmRecycleMessage(msgID, isPostOwner) {
    var d = {};
    d.msgIDs = [msgID];
    d.reason = '';

    deleteRecycleConfirmBox('RecycleMessage', d, isPostOwner);
    return false;
}

function tmDeleteMessage(msgID, isPostOwner) {
    var d = {};
    d.msgIDs = [msgID];
    d.deleteBranch = false;
    deleteRecycleConfirmBox('DeleteMessage', d, isPostOwner);
    return false;
}

function deleteRecycleConfirmBox(wsMethod, data, isPostOwner) {
    data.sendMail = false;
    qtip.confirm(ln.deleteMessageWarning,
        ln.buttonSubmitValue, (isPostOwner ? '' : ln.buttonSubmitNotifyValue), ln.buttonCancelValue,
        function () { tmCallManagementWS(wsMethod, data); },
        function () { data.sendMail = true; tmCallManagementWS(wsMethod, data); });
}

function tmCallManagementWS(method, data) {
    JQCallWebService('ws/Management.aspx/' + method, data,
    function (rtn) {
        var returnedResult = rtn.d;

        if (returnedResult === true) {

            if (method == 'DeleteMessage' && data.msgIDs[0] == currentThreadID) {
                // delete thread must redirect to forum
                self.location.replace(cPathInfo.ForumDir + 'tt.aspx?forumid=' + currentForumID.toString());
            } else {
                RefreshAfterDelete();
            }


        }

    }, JQOnCallError);

//    consoleLog(method);
//    consoleLog(data);
}

function markHelpful(msgID, self) {
    $(self).css('cursor', 'progress');
    ajaxMarkHelpfulRewardRequest(msgID, false);
}

function markAnswer(msgID, self) {
    $(self).css('cursor', 'progress');
    ajaxMarkHelpfulRewardRequest(msgID, true);
}

function ajaxMarkHelpfulRewardRequest(msgID, markAsAnswer) {
    JQCallWebService("ws/Management.aspx/MarkMessageHelpful",
        { msgID: msgID, markAsAnswer: markAsAnswer },
        function(rtn) {
            var returnedResult = rtn.d;
            if (returnedResult === true) tm.refreshButton.click();
        }, JQOnCallError);    
}

function findRewardMsg(msg) {

    var $rewardMsg = $('.rewardMsg');

    if ($rewardMsg.length>0) {
        $.scrollTo($rewardMsg);
        return;
    }

    JQCallWebService("ws/Message.aspx/FindRewardMsg", { 'msgID': msg },
        function(rtn) {
            var returnedMsgID = rtn.d;

            if (returnedMsgID == 0) return;

            if ($get(returnedMsgID)) {
                $.scrollTo($get(returnedMsgID));
            } else {
                self.location.href = cPathInfo.ForumDir + 'FindPost/' + returnedMsgID;
            }
        },
        JQOnCallError
     );

}

function listSolution(msgID, bool, self) {

    $(self).css('cursor', 'progress');

    var func = function() {
        if (!tm.inTreeMode)
            tm.refreshButton.click();
        else
            self.location.replace(cPathInfo.Url);
    };

    if (bool) {
        SaveSolutionOnlyCookie(func);
    } else {
        KillSolutionOnlyCookie(func);
    }

    return false;
    // maybe a scroll to first solution after postback is complete?
}

function scrollToFirstReply() { // called in tm.aspx.vb
    $.scrollTo($('.msgRepliesDiv').get(0));
}


function MarkFriendOrIgnore(cmd, userid, self) {

    $(self).css('cursor', 'progress');

    JQCallWebService("ws/Contact.aspx/MarkFriendOrIgnore", { command: cmd, contactID: userid },
            function (rtn) {
                var returnedResult = rtn.d;
                if (returnedResult == 'success') {
                    tm.refreshButton.click(); //refreshButton must be defined on pages that calls this method (Page that contains RadMenu)
                }
            });

}


function RegisterRatingControl() {
    
    $('select.ratingSelect')
        .rating({ showCancel: false })
        .bind('change.ajaxhandle', HandlerRatingBegin);

    $body
        .off('click.ajaxhandle', 'a.thumb')
        .on('click.ajaxhandle', 'a.thumb', HandlerRatingBegin);
    
    $('a.thumbdisabled').fadeTo(300, 0.25);
}

function HandlerRatingBegin() {
    var $ratingTool = $(this);

    var msgID = parseInt($ratingTool.attr('id').replace(/[^\d]/gi, ''));
    var rating;

    if (this.tagName.toLowerCase() == 'select') {
        rating = parseInt($ratingTool.val());
    }
    else {
        rating = ($ratingTool.hasClass('up')) ? 1 : -1;
    }

    JQCallWebService("ws/Message.aspx/RateMessage", { msgID: msgID, rating: rating },
            function (rtn) {
                var returnedVal = rtn.d;
                var successFailureFlag = returnedVal[0];
                var newRatingOption = { showCancel: false, disabled: true };
                var $resultSpan = $('#ratingResult' + msgID);
                var $_stars = $(this);
                var $_thumbs = $('#ratingThumbs' + msgID).find('a');

                if (successFailureFlag == 'fail') {

                    qtip.notice($resultSpan, ln.rateFailure,
                        { position: { target: $resultSpan, at: 'left center', my: 'right center' }, 
                            hide : {inactive : 2000} });

                    if (this.tagName.toLowerCase() == 'select') {

                        // unbind handler
                        $_stars.unbind('change.ajaxhandle');

                        // remove current
                        $_stars.next().remove();
                        $_stars.prop('hasProcessed', false); // new setting to allow re-evaluation of select

                        // disable the select

                        if ($_stars.attr('disabled') != 'disabled') $_stars.attr('disabled', 'disabled');

                        // retrieve original value

                        var originalSelectValue;

                        if ($resultSpan.children('a').size() == 0) { // not admin or mod so no link
                            originalSelectValue = $resultSpan.text();
                        } else {
                            originalSelectValue = $resultSpan.children('a').text();
                        }

                        // restart rating
                        $_stars.rating(newRatingOption);

                        //return to original value
                        $_stars.val((Math.round(originalSelectValue) - 3)).change();

                        $_stars.next().fadeTo(300, 0.25);

                    } else {

                        // disable onclick handler
                        $_thumbs.unbind('click.ajaxhandle');
                        $_thumbs.fadeTo(300, 0.25);

                    }

                    return;
                }

                // end of failed attempts

                var totalRating = parseInt(returnedVal[1]);
                var totalRateCount = parseInt(returnedVal[2]);
                var avgRating;

                if (totalRateCount == 0) totalRateCount = 1;

                if (this.tagName.toLowerCase() == 'select') {

                    // calculate the avgValue for star
                    avgRating = (totalRating / totalRateCount).toFixed(2);
                    var selectValueToSet = Math.round(avgRating) - 3;

                    // prevent repeating the ajax action
                    $_stars.unbind('change.ajaxhandle');

                    // set select to disabled
                    if ($_stars.attr('disabled') != 'disabled') $_stars.attr('disabled', 'disabled');

                    // remove current stars
                    $_stars.next().fadeOut('slow', function () {

                        $(this).remove();
                        $_stars.prop('hasProcessed', false); // new setting to allow re-evaluation of select

                        // rebuild the star; set select to avgRating;
                        $_stars.rating(newRatingOption);
                        $_stars.val(selectValueToSet).change();

                        $_stars.next().fadeTo(300, 0.25);
                    });

                } else {

                    // prevent repeating the ajax action
                    $_thumbs.unbind('click.ajaxhandle');
                    $_thumbs.fadeTo(300, 0.25);

                    // set avgRating
                    avgRating = ((totalRating > 0) ? '+' : '') + totalRating.toString();

                }

                // set text

                if ($resultSpan.children('a').size() == 0) { // not admin or mod so no link
                    $resultSpan.text(avgRating);
                } else {
                    $resultSpan.children('a').text(avgRating);
                }

                //setting rate count to the div
                $('#ratingDiv' + msgID).attr('title',
                    String.format(((totalRateCount > 1) ? ln.tmRateCount_pl : ln.tmRateCount_sing), totalRateCount));
                    
            },
            JQOnCallError, this // this refers to the event target passed to as context of the Ajax framework
        );
}




function ProduceSearchPhraseFromString(str, top, useAnd) {
    var cleanedTitle = str.trim().replace(/[^\w\s\.\-]/gi, '');
    var arrToRemove = cleanedTitle.split(' ');

    arrToRemove.sort(function (a, b) { return (b.length - a.length); });

    if (!top) top = 4;

    var arrLongestWords = new Array();

    for (var j = 0; j < arrToRemove.length; j++) {
        if (arrLongestWords.indexOf(arrToRemove[j]) == -1) {
            arrLongestWords.push(arrToRemove[j]);

            if (arrLongestWords.length == (arrToRemove.length < top ? arrToRemove.length : top))
                break;
        }
    }

    var arrFresh = cleanedTitle.split(' ');
    var arrFinal = new Array();

    for (var i = 0; i < arrFresh.length; i++) {
        var currentWord = arrFresh[i];
        if (arrLongestWords.indexOf(currentWord) != -1 && arrFinal.indexOf(currentWord) == -1) {
            if (currentWord.match(/[^\w]/gi)) currentWord = '"' + currentWord + '"';
            arrFinal.push(currentWord);
        }

    }

    return arrFinal.join((useAnd ? ' ' : ' OR ')).trim();
}

function getSearchTermSuggestionFromWiki(phrase, onComplete) {
    $.ajax(
        { url: "http://en.wikipedia.org/w/api.php",
            dataType: "jsonp",
            data: { action: "opensearch",
                search: phrase,
                format: "json"
            },
            success: function (data, textStatus, xhr) {
                onComplete(data[1]);
//                $("#results").empty();
//                $.each(data[1], function (_, result) {
//                    $("#results").append("<li>" + result + "</li>");
//                });
            },
            error: function (xhr, textStatus, errorThrown) {
                onComplete(errorThrown);
            }
        }); 
    
}
var _searchSuggest = '';

function getSearchTermSuggestion(phrase, onComplete) {
    if (_searchSuggest!= phrase.trim()){
        JQCallWebService("ws/Search.aspx/SearchSuggestion",
                { searchPhrase: phrase },
                function (rtn) {
                    var returnedData = rtn.d;
                    _searchSuggest = phrase;
                    onComplete(returnedData);
                });
        }
}

function initiateRelatedTopicSearch(phrase, forumSpec, onComplete, topN) {
    JQCallWebService("ws/Search.aspx/SearchRelatedTopics",
        {
            sRequest: {
                Phrase: phrase.trim(),
                ForumIDs: forumSpec,
                TopNResults: topN,
                Precision: ['Normal']
            }
        },
        onComplete,
        JQOnCallError);
}

function initiateSearchSimilarThreads(phrase, forumSpec, onComplete, topN, searchPrecision) {

    if (typeof searchPrecision == 'undefined') searchPrecision = ['Exact'];
    
    JQCallWebService("ws/Search.aspx/SearchSimilarThreads",
        {
            sRequest: {
                Phrase: phrase.trim(),
                ForumIDs: forumSpec,
                TopNResults: (!topN ? 30 : topN),
                Precision: searchPrecision
            }            
        },
        onComplete,
        JQOnCallError);

}

function initiateSearchRegular(phrase, forumIDs, threadID, topN, phraseIn, resultAs,
                                searchPrecision, onComplete) {
        
    // resultAs: TopicsOnly RepliesOnly Both Combined
    // searchPrecision: Fuzzy Normal Exact ExactOnly
        
    if (typeof topN == 'undefined') topN = 300;
    if (typeof resultAs == 'undefined') resultAs = 'Combined';
    if (typeof searchPrecision == 'undefined') searchPrecision = 'Exact';

    JQCallWebService("ws/Search.aspx/BeginSearch",
        {
            sRequest: {
                Phrase: phrase,
                ForumIDs: forumIDs,
                ThreadID: threadID,
                TopNResults: topN,
                ListResultAs: resultAs,
                Precision: searchPrecision,
                PhraseFoundIn: phraseIn
            }
        },
        onComplete,
        JQOnCallError);

}

function searchCheckIfComplete(aid, initialSearchTime, onComplete) {

    var data = { theAID: aid, beginTime: initialSearchTime };

    JQCallWebService("ws/Search.aspx/CheckIfStepOneCompleted", data, onComplete, JQOnCallError);
}


function beginRetrieveingSimpleSearchResults(searchID, onBeforeRetrieving, onFinishRetrieving) {
    onBeforeRetrieving();

    var data = { searchID: searchID, top : -1 };

    JQCallWebService("ws/Search.aspx/RetrieveSimpleSearchResults", data, onFinishRetrieving, JQOnCallError);

}


$doc.ready(function () {
    registerSubscriptionMenuHover();
});

var _subMenuLinkSel = '#subscriptionMenuLink';
var _subsDataKey = 'currentlevel';
var _subMenuDatakey = 'sublevel';

function registerSubscriptionMenuHover() {
    var $subMenuLink = $(_subMenuLinkSel);

    $subMenuLink.qtip({
        overwrite: false,
        content: { text: $('#subOptions').html() },
        position: { my: 'top left', at: 'bottom left', adjust: { y: 5, x:-15} },
        show: { event: ($subMenuLink.data(_subsDataKey) != -1 ? 'mouseenter' : 'click') },
        hide: { fixed: true, delay: 600, event: 'unfocus mouseleave' },
        style: { width: '250px', tip: false, classes: 'qtip-apgsmallmenu qtip-shadow' }
    });
}

function HandleSubscription(self, isForum, id) {
    var currentLevel = $(self).data(_subsDataKey);

    if (currentLevel != -1) {
        return SetSubscription(self, -1, isForum, id);
    } else {
        return false;
    }
}

function SetSubscription(self, action, isForum, id) {
    var $link = $(self);

    $link.css('cursor', 'progress');
    $('.subOption').removeClass('bold');

    var subscribeLinkText = (isForum ? ln.ttOptionSubscribeToForum : ln.tmOptionSubscribeToThread);
    var unsubscribeLinkText = (isForum ? ln.ttOptionUnSubscribeToForum : ln.tmOptionUnSubscribeToThread);

    JQCallWebService(
        "ws/Subscription.aspx/SubscribeTo" + (isForum ? 'Forum' : 'Thread'),
        { id: id, action: action },
        function () {

            setTimeout(function () { $link.css('cursor', 'pointer'); }, 200);

            var $subMenuLink = $(_subMenuLinkSel);
            var qtipapi = $subMenuLink.data('qtip');

            if (action != -1) {
                $subMenuLink.text(unsubscribeLinkText);
                qtipapi.set('show.event', 'mouseenter');
            } else {
                $subMenuLink.text(subscribeLinkText);
                qtipapi.set('show.event', 'click');
            }

            setTimeout(function () { qtipapi.hide(); }, (action != -1? 500 : 150));
            
            $subMenuLink.data(_subsDataKey, action);

            // from drop down - instead of from menu link
            if ($link.hasClass('subOption')) $link.addClass('bold');
        }

    );

    return false;
}
// Tree loading

var treeDivASelector = 'div#treeIframeDiv a.msg';
var treeDivSelector = 'div#treeIframeDiv';
var currentRequestTreeMsgID = 0;

$(function() {
    if (tm.inTreeMode) {
        setTreeDivResizable();
        setPrevNextLinkHandler();
        $('#' + tm.msgTreeHiddenField).val(currentThreadID); // this is required otherwise on initial load the hidden field is empty
    }
});

$(function () {

    setTimeout(function () { // setTimeout 100ms required because server side AddResponseScript/Format happens later
        if (tm.inTreeMode) {

            if (currentRequestTreeMsgID == 0) { loadTree(currentThreadID, currentThreadID); } // load tree regular mode

            if (currentRequestTreeMsgID != 0) performFindMsg(currentRequestTreeMsgID, false); //find msg mode, ie when tree m is specified

            $(treeDivSelector).delegate('a.msg', 'click', treeMsgClick);
            
            setTimeout(function () { // need delaying, as append is kind of slow in loadTree
                $(treeDivASelector + ':first').addClass('bold');
                readyToPerformScroll = true;

            }, 1000);

            setScrollHandler();
        }
    }, 100);
});

function performFindMsg(msgID, rebindScrollHandler) {

    loadTree(msgID, currentThreadID, null, true);
    //console.log('loadTree');
    
    //click to load the first msg
    setTimeout(function() { $(treeDivSelector + ' a#treelink' + msgID.toString()).click(); }, 1500);
    //console.log('first setTimeout');
    
    //load the tree above the clicked msg
    setTimeout(function () { loadTree(msgID, currentThreadID, null, false, true); }, 1800);
    //consoleLog('ln52'); 
    //console.log('second setTimeout');
    
    if (rebindScrollHandler === true) { // this is when we find msg after delete
        //console.log('rebindScrollHandler === true');
        dontCheckScrollDown = false;
        dontCheckScrollUp = false;
        setScrollHandler();
    }
}

function setTreeDivResizable() {
    var $treeContainer, currentHeight, currentWidth;
    $treeContainer = $('#' + tm.treeIframeContainment);

    if ($treeContainer.size() == 0) return;

    currentHeight = $treeContainer.height();
    currentWidth = $treeContainer.width();

    $treeContainer.resizable(
        {
            alsoResize: treeDivSelector,
            distance: 10,
            delay: 20,
            containment: '#main',
            minHeight: currentHeight,
            maxHeight: $win.height() / 2,
            maxWidth: currentWidth, minWidth: currentWidth,
            start: function(e, ui) {
                readyToPerformScroll = false;
            },
            stop: function(e, ui) {
                setTimeout(function() {
                    readyToPerformScroll = true;
                }, 2000);
            }
        }
     );

    setDivResizingBehavior();
        
}

function setDivResizingBehavior() {
    var tempCounter = 0;
    var windowResizeTimeout;

    window.onresize = function() {
        window.clearTimeout(windowResizeTimeout);
        windowResizeTimeout = window.setTimeout(DelayedResize, 100);
    };

    /*
    elaborated windows Resize test to enure consistent cross browser behavior
    */
    var sizeBeforeResize = 0;
    var $iframeContainment = $('#' + tm.treeIframeContainment);
    var iframeSizeDiff = $iframeContainment.width() - $(treeDivSelector).width();

    function DelayedResize() {
        window.status = ++tempCounter;
        // Add your original window.onresize implementation here

        var mainWidth = $('div#main').children('div:first').width();

        if (mainWidth != sizeBeforeResize) { // size is changed on the resize event

            setTimeout(function() {
                // set new minwidth and maxwidth for the resizable element due to change in window size
                $iframeContainment.resizable("option", "minWidth", mainWidth)
                            .resizable("option", "maxWidth", mainWidth);
            }, 100);

            setTimeout(function() {
                // resize the resizable element
                $iframeContainment.animate({ width: mainWidth.toString() + 'px' }, 200);
            }, 400);

            setTimeout(function() {
                //resize the div and the iframe
                $(treeDivSelector).animate(
                            { width: ($iframeContainment.width() - iframeSizeDiff).toString() + 'px' }, 200);
            }, 700);

            sizeBeforeResize = mainWidth;
        }

    }
}

function setPrevNextLinkHandler() {
    $('a#treePrevLink').click(function(e) { treeNavigateTo('prev'); e.preventDefault(); });
    $('a#treeNextLink').click(function(e) { treeNavigateTo('next'); e.preventDefault(); });
    $('a#treeReloadLink').click(function(e) { treeReload(); e.preventDefault(); });
}

var readyToPerformScroll = false;

var newScrollTimeout;

function setScrollHandler() {
    $(treeDivSelector).unbind('scroll').scroll(function() {
        var _self = this;
        window.clearTimeout(newScrollTimeout);
        newScrollTimeout = setTimeout(function() { checkScroll($(_self)); }, 400);
    });
}

var dontCheckScrollDown = false;
var dontCheckScrollUp = false;

function checkScroll($treeDiv) {
    // this function performs 2 checks
    // 1) when scrolled to bottom, check if 'last' is present and if so, auto fetch
    // 2) when scrolled to top, check if 'first' is present and if NOT, auto fetch
    // if while fetching, we do nothing

    //console.log('checkScroll');
    
    if (!readyToPerformScroll || stillInAjaxProgress) return false;

    if (!dontCheckScrollDown && $treeDiv[0].scrollHeight - $treeDiv.height() - $treeDiv.scrollTop() <= 14) {
        
        var $lastLink = $treeDiv.find('a.last');

        if ($lastLink.size() != 0) {
            fetchNextTreePage(parseInt($lastLink.attr('id').replace('treelink', '')), $lastLink); // supply jqTarget to rid of the last on existing link, simulating click
        } else {
            dontCheckScrollDown = true;
        }
    }

    else if (!dontCheckScrollUp && $treeDiv.scrollTop() <= 14) {

        var $firstLink = $treeDiv.find('a.msg:first');

        if (!$firstLink.hasClass('first')) {
            signalTreeState('loading');
            $treeDiv.css({'overflow':'hidden', 'visibility' : 'hidden'});
            loadTree(parseInt($firstLink.attr('id').replace('treelink', '')), currentThreadID, null, false, true);
            //consoleLog('ln192'); 
        } else {
            dontCheckScrollUp = true;
        }
    }

    if (dontCheckScrollUp && dontCheckScrollDown) {
        $(treeDivSelector).unbind('scroll');
    }
    
}

var stillInAjaxProgress = false;

function loadTree(msgID, threadID, jqTarget, findmsg, prepend) {
    /*
    We are treating the last and first designation differently. when there is a class='last' link, 
    it means we can still try to find if there is more links in the thread. If there is no class='last', then we reached the end.
    
    the 'first' designation is only given to the first post. when there is no first in the entire link list, then we can go look for
    "previous page" in the tree.
    
    jqTarget is only used when fetching next page.
    */
    
    stillInAjaxProgress = true;

    JQCallWebService("ws/Message.aspx/GetTree",
        { 'msgID': msgID, 'threadID': threadID, 'findmsg': (findmsg ? findmsg : false), 
            'direction': (prepend === true ? 1 : 0)},
    function (rtn) {
        var returnedVal = rtn.d;

        var finalLinksHTML = buildTreeHTML(returnedVal);

        if (jqTarget && prepend !== true) jqTarget.removeClass('last'); // there can be more so the 'last' designation is gonna change

        if (finalLinksHTML != '') {
            var $treeDiv = $(treeDivSelector);

            if (prepend === true) $treeDiv.prepend(finalLinksHTML); else $treeDiv.append(finalLinksHTML);
            if (prepend !== true && returnedVal.length == tm.tpg) $(treeDivASelector + ':last').addClass('last'); //there is still possibility to have more pages

            if (prepend === true) {
                // scroll the found msg into view
                setTimeout(function () {
                    var $theFirstLinkBeforeFetch = $(treeDivSelector + ' a#treelink' + msgID.toString());
                    $treeDiv.scrollTo($theFirstLinkBeforeFetch, 1000, { margin: true });
                    //console.log('new tree scroll');
                    $theFirstLinkBeforeFetch.focus();
                }, 1000);
            }

            setTimeout(function () {
                $treeDiv.css({ 'overflow': 'auto', 'visibility': 'visible' });
                performDateFormat(false);

                JQCallWebService('ws/TrackRead.aspx/GetThreadLastRead', { threadID: threadID }, function (r) { // check read
                    var referenceDate = new Date(r.d);
                    $treeDiv.find('.checkDate span.performdateformat').each(function () {
                        var $this = $(this);
                        var thisdate = new Date($this.data('date') || cMemberInfo.lastVisit);

                        if (thisdate.getTime() - referenceDate.getTime() > 0) {
                            var id = $this.parent().data('trackmid');
                            if (!$('#treelink' + id).hasClass('bold'))
                                $('#newimg_' + id).attr('src', pageThemeImageURL + 'newestmsg.gif');
                        }
                    });
                });

            }, 1100);
        }

        setTimeout(function () { stillInAjaxProgress = false; }, 2000);

    }, JQOnCallError);
}

function buildTreeHTML(listdata) {
    var linkTemplate = "<div class='yMargin5' style='margin-left:{1}px;'> <img src='{2}' /> " +
    "<a href='#' id='treelink{0}' class='msg {12}' style='{10}' data-trackmid='{0}'>{3}</a> <img src='{14}blank.gif' id='newimg_{0}' />" +
    "{9} {11} {13} " +
    "<span class='lMargin5 small checkDate' data-trackmid='{0}'>{8} <a href='showprofile.aspx?memid={4}' target='_blank' {7}>{5}</a> - {6}</span>" +
    "</div>";

    var totalLinks = [];

    for (var i = 0; i < listdata.length; i++) {
    
        var currentMsg = listdata[i];
        //	messageID	llevel	subject msgIcons	dateCreated	login	mem	Ignored isAnswer isRewarded moderated Deleted
        
        totalLinks[totalLinks.length] = String.format(linkTemplate,
            currentMsg.messageID, //0
            15 + 20 * parseInt(currentMsg.llevel), // 1
            pageThemeImageURL + 'mIcons/m' + currentMsg.msgIcons + '.gif', // 2
            currentMsg.subject.replace(/</g, '&lt;').replace(/>/, '&gt;'), //3
            currentMsg.mem, //4
            currentMsg.login, //5
            currentMsg.dateCreated, //6
            ((currentMsg.Ignored == '1') ? 'class="ignored" title="' + ln.ttMemberIgnored + '"' : ''), //7
            ln.byDesc,//8
            ((currentMsg.isRewarded == 1) ? '<span class="msgState ttanswered">' + ln.tmIsAnswer + '</span>' : ((currentMsg.isAnswer == 1) ? '<span class="msgState tthelpful">' + ln.tmIsHelpful + '</span>' : '')), //9
            ((currentMsg.Deleted == 1) ? 'text-decoration: line-through;' : ''), //10
            ((currentMsg.Moderated == 1) ? '<span class="msgState moderatedMsg">' + ln.tmMessageRequireModerationDesc + '</span>' : ''), //11
            ((currentMsg.llevel == 0) ? 'first' : ''), //12
            ((currentMsg.HasAttachment == 1) ? '<img src="' + pageThemeImageURL + 'attachment.gif" />' : ''), //13
            pageThemeImageURL
            );
    }

    return totalLinks.join('');
}

function treeMsgClick(e) {
    
    //consoleLog('treeMsgClick');
    
    var $target = $(this);

    $(treeDivASelector + '.bold').removeClass('bold');

    $target.addClass('bold');

    var currMsgID = parseInt($target.attr('id').replace('treelink', ''));

    $('#' + tm.msgTreeHiddenField).val(currMsgID);
        
    $target.focus();

    tm.refreshButton.click();

    if ($target.hasClass('last')) fetchNextTreePage(currMsgID, $target);

    // mark message read:
    var id = $target.data('trackmid');
    $('#newimg_' + id).attr('src', pageThemeImageURL + 'blank.gif');
    
    return false;
}

function treeNavigateTo(dir) { // handler for the previous / next link on tm.aspx

    var $currMsgLink = $(treeDivASelector + '.bold:first');

    if (dir == 'next') {

        var $next = $currMsgLink.parent().next().find('a.msg');
        
        if ($next.length == 0) {
            signalTreeState('last');
        } else {
            $next.click();
        }

    } else {

        var $prev = $currMsgLink.parent().prev().find('a.msg');

        if ($prev.length == 0) {
            signalTreeState('first');
        } else {
            $prev.click();
        }
    }
    
}

function treeReload(msgID) {
    //consoleLog('treeReload ' + msgID);
    
    if (isNaN(msgID)) {
        msgID = parseInt($(treeDivASelector + '.bold:first').attr('id').replace('treelink', ''));
    }

    signalTreeState('loading');
    $(treeDivSelector).html('').css({'visibility' : 'visible' });
    performFindMsg(msgID, true);
}

function fetchNextTreePage(currentMsgID, jqTarget) {
    loadTree(currentMsgID, currentThreadID, jqTarget);
    //consoleLog('ln359'); 
}

function signalTreeState(state) {

    var signal = '<div class="endOfTreeNotice"></div>'
    var finalStringSelector = 'div#treePrevNextLinks div.endOfTreeNotice';
    var treePrevNextLinkSelector = 'div#treePrevNextLinks';

    if ($(finalStringSelector).size() == 0) {
        $(signal).prependTo(treePrevNextLinkSelector);
    }

    var $finalString = $(finalStringSelector);

    $finalString.text(function() {
        return ((state == 'first') ? ln.tmReachFirstInTreeDesc : ((state == 'last') ? ln.tmReachLastInTreeDesc : ln.loadingDesc));
    })
    .css({
        'top': function() {
            var $treeDiv = $(treeDivSelector);
            var finalTop = '-' + ($(treePrevNextLinkSelector).offset().top - ($treeDiv.offset().top + $treeDiv.height()) + ($treeDiv.height() / 2)).toString() + 'px';
            return finalTop;
        }
    ,
        'left': function() {
            var finalLeft = ($(treeDivSelector).width() - $finalString.width()) / 2;
            return finalLeft.toString() + 'px';
        }
    })
    .fadeIn('normal', function() { setTimeout(function() { $finalString.fadeOut(); }, ((state == 'loading') ? 750 : 1500)); });
}

function treeProcessAfterDelete() {

    var $firstLinkInTree = $(treeDivASelector).eq(0);
    
    if ($firstLinkInTree.hasClass('bold') && $firstLinkInTree.hasClass('first')) { // topic message deleted
    
        self.location.replace(cPathInfo.ForumDir + 'tt.aspx?forumID=' + currentForumID.toString());
        //console.log('self.location.replace');
    } else {

        var $prevLink = $(treeDivASelector).filter('.bold').parent().prev().children('a.msg');
        var msgID = parseInt($prevLink.attr('id').replace('treelink', ''));
        //console.log('deleted msgID = ' + msgID.toString());
        treeReload(msgID);
    }
    
}
var _previewContent, _previewInterval, _previewIntervalMs = 5000;

function openPreview(manual) {

    var upfileValue = _.isUndefined(window.postingInterface) ? '' : window.postingInterface.attInfo.attachmentKey;

    var msgContent = getEditorContent();
    var editorDim = getEditorDimension();

    var $previewDiv = $('#previewdiv');
    var $previewContent = $previewDiv.find('.previewcontent');

    if (manual) {
        $.scrollTo($previewDiv);
        $previewContent.show();
    }

    if (_previewContent == msgContent || (!manual && !$previewContent.is(':visible'))) return false;

    window.clearInterval(_previewInterval);

    JQCallWebService("ws/Message.aspx/Preview",
        { message: msgContent, attachmentID: upfileValue, forPage: getForPage() },
        function (r) {

            $previewContent.html(r.d);
            $previewContent.css({ 'max-height': editorDim.h, 'padding': '10px' });
            $previewDiv.css('width', editorDim.w);
            $previewDiv.fadeIn();

            if (manual) {
                $.scrollTo($previewDiv);
            }
            //$previewContent.scrollTo({ top: 0, left: 0 }, { duration: 0 });
            //$previewContent.effect('highlight', {}, 1500);

            var imgTmpl = '<img src="{0}download.axd?file={1};{2}&where={3}&r={4}" />';

            var $allAttachedImg = $previewContent.find('span.mceNonEditable');

            for (var i = 0; i < $allAttachedImg.length; i++) {
                var $attached = $allAttachedImg.eq(i);
                var attachedText = $attached.text();
                var imgIndex = window.postingInterface.attachmentsArray[attachedText];
                $attached.replaceWith(
                    String.format(imgTmpl, cPathInfo.ForumDir, imgIndex, upfileValue,
                    (wysiwygHelper.forPage == 1 ? 'msg' : (wysiwygHelper.forPage == 2 ? 'pm' : 'event')), attachedText));
            }

            _previewContent = msgContent;
            startPreviewInterval();
        });

    return false;
}

function getForPage() {
    if (typeof tinyMCE != 'undefined') { //WYSIWYG
        return wysiwygHelper.forPage;
    } else { // textbox
        return wysiwygAsTextarea.forPage;
    }    
}

function getEditorDimension() {
    var editorWidth, editorHeight;
    
    if (typeof tinyMCE != 'undefined') { //WYSIWYG
        var $mceLayout = $('table.mceLayout');
        editorWidth = $mceLayout.width();
        editorHeight = $mceLayout.find('td.mceIframeContainer').height();

    } else { // textbox
        var $editor = $(wysiwygAsTextarea.TextareaSelector);
        editorWidth = $editor.width();
        editorHeight = $editor.height();
    }

    return { w: editorWidth, h: editorHeight };
}

function regPreviewToggleClick() {
    $('#previewdiv').on('click', 'a.toggler', function () {

        var $link = $(this);
        var $previewContent = $('div.previewcontent');

        $previewContent.slideToggle('fast', function () {
            var isContentOpen = $previewContent.is(':visible');
            var innerHtml = ln.buttonPreviewValue + '&nbsp;' +
                (isContentOpen ? ln.entUpArrow : ln.entDownArrow);
            $link.html(innerHtml);
            
            if (isContentOpen) openPreview(false); // not from the buttons, so false
        });

        return false;
    });
}

var _previewIni = false;
var _previewAutoStart = false;

function displayPreviewDiv() { // called in NewRTECustomJS.js
    if (_previewIni) return;
    var $previewDiv = $('#previewdiv');
    var editorDim = getEditorDimension();

    $previewDiv.css('width', editorDim.w).fadeIn();
    _previewContent = ''; // prevent sending of blank initially to preveiw

    if (_previewAutoStart) {
        setTimeout(function () { $previewDiv.find('a.toggler').trigger('click'); }, _previewIntervalMs);
        startPreviewInterval();   
    }

    _previewIni = true;
}

function startPreviewInterval() {
    _previewInterval = setInterval(function () { openPreview(false); }, _previewIntervalMs);
}

$doc.ready(function () {
    regPreviewToggleClick();

    if (typeof wysiwygAsTextarea != 'undefined') { // tinyMCE's focus handler defined in NewRTECustomJS
        $(wysiwygAsTextarea.TextareaSelector).one('focus', function () {
            displayPreviewDiv();
        });
    } else {
        var previewIni =
        setInterval(function () {
            if (typeof tinyMCE != 'undefined') {

                tinyMCE.activeEditor.onMouseUp.add(displayPreviewDiv);
                tinyMCE.activeEditor.onKeyDown.add(displayPreviewDiv);
/*
                tinyMCE.activeEditor.onChange.add(function () {
                    openPreview(false);
                });
*/

                setTimeout(function () {
                    window.clearInterval(previewIni);
                }, 500);

            }
        }, 1000);

    }
});

$doc.ready(function () {
    api_init();
});

var _api_init = false;

function api_init(forcedExec) {
    if (!forcedExec && _api_init) return;
    api_registerSignupbox();
    api_registerAlternativeLoginLinks();
    api_registerPollIndividualVote();
    //consoleLog('api_init');
    _api_init = true;
}

var _onsiteApi_qtipNoticeOption = { position: { at: 'middle right', my: 'middle left', adjust: { x: 2} }, hide: { inactive: 3000} };

function api_registerPollIndividualVote() {
    var $surveyDiv = $('div.api_surveybox');
    
    if (!$surveyDiv[0]) return;
    
    _api_loadSurvey($surveyDiv);

    $body.on('click', '.castvote', function () {
        var $btn = $(this);
        var $api_box = $btn.closest('.api_surveybox');

        var $polls = $api_box.find('div.voteInterface');
        var atLeastOneSubmitted = false;
        var submittedCount = 0;
        for (var i = 0; i < $polls.length; i++) {
            var $poll = $polls.eq(i);

            var $checkedBoxes = $poll.find('input[type=checkbox]:checked');
            var $radio = $poll.find('input[type=radio]:checked');
            var votes = [];

            if ($checkedBoxes[0]) {
                $checkedBoxes.each(function () {
                    votes.push($(this).val());
                });
            } else if ($radio[0]) {
                votes.push($radio.val());
            }

            if (votes.length == 0) continue;

            submittedCount += 1;

            atLeastOneSubmitted = true;

            JQCallWebService('ws/Poll.aspx/CastVote', { pollID: $poll.data('pollid'), votes: votes }, function (r) {

                if (r.d) {

                }
            }, JQOnCallError);

        }

        if (atLeastOneSubmitted) {
            $btn.effect('highlight', submittedCount * 400);
            
            setTimeout(function () {
                _api_loadSurvey($surveyDiv);
                $btn.hide();
            }, submittedCount * 300);
        }

        return false;
    });
}

function _api_loadSurvey($surveyDiv) {
    var pollids = $surveyDiv.data('pollids');

    $('<div />').load(cPathInfo.ForumDir + 'ws/Poll.aspx?pollid=' + pollids + ' #AllPolls div', function (r) {
        var $div = $(this);

        $div.find('.voteResultRow, .voteBtnRow').remove();
        $div.find('.voteresultHead').removeClass('center');
        $div.find('.voteresults td').css('border-bottom', '1px dashed #DDD');

        $surveyDiv.find('div:first').replaceWith($div);

        if ($surveyDiv.find('input')[0])
            $surveyDiv.find('.castvote').show();

        $surveyDiv.show();

    });    
}

function api_registerSignupbox() {
    var $signupDiv = $('div.api_signupbox');
    
    if (!$signupDiv[0]) return;

    if (cMemberInfo.memID != -1) {
        $signupDiv.remove();
        return;        
    } else {
        $signupDiv.show();
    }

    var $1stsignupdiv = $signupDiv.eq(0);

    var $name = $1stsignupdiv.find('input.signup_name');
    var $email = $1stsignupdiv.find('input.signup_email');
    var $pass = $1stsignupdiv.find('input.signup_pass');
    var $cptaimg = $1stsignupdiv.find('img.signup_cptaimg');
    var $cpta = $1stsignupdiv.find('input.signup_cptainput');
    var $signupbutton = $1stsignupdiv.find('.signup_button');

    $cpta.focus(function () {
        _api_resetCaptcha($cptaimg, $cpta);
    });

    $signupbutton.click(function () {

        var neededFields = [$name, $email, $pass, $cpta];

        if (_api_requiredFieldFailedNotice(neededFields)) return false;

        var data = {};
        data.username = $name.val();
        data.email = $email.val();
        data.pass = ($pass[0] ? $pass.val() : '');
        data.captchaVal = ($cpta[0] ? $cpta.val() : '');

        JQCallWebService('ws/Login.aspx/RegisterUser', data, function (r) {
            //{.Success, .SuccessResponse, .FailedReason, .NewMemberID, .RegStatus, .RegisterResult}
            var regResult = r.d;

            if (regResult.RegisterResult == 0 && regResult.NewMemberID > 0) {

                if (regResult.SuccessRedirectTo != '') {
                    self.location.href = regResult.SuccessRedirectTo;
                    return true;
                }

                var handleOk;

                if (regResult.RegStatus == 0) {
                    handleOk = function () {
                        self.location.reload();
                    };

                } else {
                    handleOk = function () {
                        _api_blankOutAllInput(neededFields);
                    };
                }

                qtip.confirm(regResult.SuccessResponse, ln.buttonSubmitValue, '', '', handleOk);

            } else {
                var $noticeEle = (regResult.RegisterResult == 1 ? $name :
                    (regResult.RegisterResult == 2 ? $email : (regResult.RegisterResult == 3 ? $signupbutton : $cpta)));

                qtip.notice($noticeEle, regResult.FailedReason, _onsiteApi_qtipNoticeOption);

                if ($noticeEle.prop('tagName') == 'INPUT') {
                    $noticeEle.val('');
                };

            }

            _api_resetCaptcha($cptaimg, $cpta);

        });

        return true;
    });

    $signupDiv.filter(':gt(0)').remove();
}

function _api_requiredFieldFailedNotice($fields) {
    var failed = false;
    for (var i = 0; i < $fields.length; i++) {
        if ($fields[i][0] && $fields[i].val().trim() == '') {
            failed = true;
            qtip.notice($fields[i], ln.warnRequiredDesc, _onsiteApi_qtipNoticeOption);
        }
    }
    return failed;
}

function _api_resetCaptcha($img, $input) {
    $img.attr('src', pageThemeImageURL + 'CaptchaImage.axd?p=' + randomString(5));
    $input.val('');
}

function _api_blankOutAllInput($fields) {
    for (var i = 0; i < $fields.length; i++) {
        if ($fields[i][0]) {
            $fields[i].val('');
        }
    }
}

function api_registerAlternativeLoginLinks() {
    var $topLoginLink = $('#subnav-login');

    if (!$topLoginLink[0]) return;

    $('a.clickToOpenLogin')
        .html($topLoginLink.html())
        .click(function () {
            $topLoginLink.triggerHandler('click'); //,
            //[$(this), { position: { my: 'bottom right', at: 'top right', adjust: { y: -15}}}]);
            return false;
        }
    );
}

function CreateMemberMenu($dataLink, $div) {
    var isownpost = $dataLink.data('isownpost'),
        login = $dataLink.data('login'),
        isfriend = $dataLink.data('isfriend'),
        isignored = $dataLink.data('isignored'),
        viewerisguest = $dataLink.data('viewerisguest'),
        showpmlink = $dataLink.data('showpmlink'),
        isrecycled = $dataLink.data('isrecycled'),
        canTempBan = $dataLink.data('tempban'),
        messageID = $dataLink.data('messageid'),
        ip = $dataLink.data('ip'),
        mem = $dataLink.data('mem'),
        isguest = (mem == -1);

    var str = '<ul class="titleHeadInfoList">';

    var friendBlockTempl = '<li><a href="#" onclick="MarkFriendOrIgnore(\'{2}\', {0}, this); return false;">{1}</a></li>';

    if (!isownpost && !viewerisguest && !isfriend && !isguest)
        str += String.format(friendBlockTempl, mem, ln.tmMenuAddFriend, 'friend');

    if (!isownpost && !viewerisguest && !isguest) {

        var blockItemText = isignored ? ln.tmMenuUnblock : ln.tmMenuBlock;
        var blockItemCommand = isignored ? 'unignore' : 'ignore';

        str += String.format(friendBlockTempl, mem, blockItemText, blockItemCommand);        
    }

    if (!isownpost && !viewerisguest && showpmlink && !isguest) {
        var gotoURL = cPathInfo.ForumDir + 'pmsend.aspx?toMemId=' + mem;
        var onclick = '';
        
        if (cMemberInfo.popupPosting) {
            onclick = String.format('popRadWin(\'{0}&pop=true\'); return false;', gotoURL);
        }

        str += String.format('<li><a href="{0}" onclick="{2}">{1}</a></li>', gotoURL, ln.tmMenuSendPM, onclick);
    }

    if (!isignored && !isrecycled && !isguest)
        str += String.format('<li><a href="{0}">{1}</a></li>',
            cPathInfo.ForumDir + 'posts/' + escape(login), ln.profRecentPosts);

    if (canTempBan && mem != 0 && !isguest)
        str += String.format('<li><a href="javascript:void(popRadWin(\'tempban.aspx?messageID={0}\', 600, 450))">{1}</a></li>',
            messageID, ln.tmMenuTempBan);

    if (ip)
        str += String.format('<li>&nbsp;</li><li class="alignRight">{0}</li>', ip);

    str += '</ul>';

    $div.append(str);

    $div.on('click', 'a', function () {
        setTimeout(function() { $div.remove(); }, 100);
    });
}

$doc.ready(function () {
    setupBottomBreadCrumb();

    if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;

    var totalSize = $('div.breadcrumb a').size();
    // select the breadcrumb a items first, and on mouseover of the links, produce
    // the lorder value from the select menu

    if (totalSize <= 1) return; // error handling when only All Forums is present

    var timeout;

    $('div.breadcrumb').on('mouseenter', '.breadcrumbitem', function () {
        var $link = $(this);

        if (!$link.data('dropdown')) return false;

        timeout = setTimeout(function () {
            var theLinkText = $link.text().trim();

            var $theSelectMenuOptions = $('select[id$=ForumJumpMenu]').children('option');

            //now first of all, get the lorder
            var theLorderForCurrent;

            $theSelectMenuOptions.each(function () {
                if (this.text.trim().endsWith(theLinkText)) {
                    theLorderForCurrent = $(this).attr('lorder');
                    return false;
                }
            });

            if (typeof theLorderForCurrent == "undefined") return true; // error handling

            //now collect a list of related by take away last 3 digits, and find those who starts the remaining and with same length;
            var parentLorder = theLorderForCurrent.substr(0, (theLorderForCurrent.length - 3));

            var $theOptionsCollections = $theSelectMenuOptions.filter(function (i) {
                var $option = $(this);
                var lorder = $option.attr('lorder');
                if (typeof lorder == "undefined") return false;
                return (lorder.startsWith(parentLorder) && lorder.length == theLorderForCurrent.length && !$option.get(0).text.endsWith(theLinkText));
            });

            if ($theOptionsCollections.length == 0) return false;

            $link.qtip({
                overwrite: false,
                content: { text: createLinksFromOptions($theOptionsCollections) },
                position: { my: 'top left', at: 'bottom left', adjust: { y: 5} },
                show: { event: 'mouseenter', ready: true },
                hide: { fixed: true, delay: 200, event: 'mouseleave' },
                style: { tip: false, classes: 'qtip-apgsmallmenu qtip-shadow' }
            });

        }, 100);

    }).on('mouseleave', '.breadcrumbitem', function () { clearTimeout(timeout); });

});

function createLinksFromOptions($options) {
    var val = '<ul class="noneList margin0">';

    for (var i = 0; i < $options.length; i++) {
        var link = $options.get(i);
        val += '<li>' + String.format('<a class="postButtonDropdown" href="{0}">{1}</a></li>', 
            link.value, link.text.replace(/^(\-\s)+/gi, ''));
    }

    return val + '</ul>';
}

function setupBottomBreadCrumb() { // this is using add_load due to the use of drop down menu on the breadcrumb;
    $('#bottomItemsConsolidator').before($('div.breadcrumb:first').parent().html()).css('min-height', '30px');
    $('div.breadcrumb:last').addClass('altItem').css({ 'padding': '8px 10px', 'border-bottom': '0' });
}
(function (c) { c.fn.rating = function (h) { var d = { showCancel: !0, cancelValue: null, cancelTitle: "Cancel", startValue: null, disabled: !1 }, e = { hoverOver: function (a) { a = c(a.target); a.hasClass("ui-rating-cancel") ? a.addClass("ui-rating-cancel-full") : a.prevAll().addBack().not(".ui-rating-cancel").addClass("ui-rating-hover") }, hoverOut: function (a) { a = c(a.target); a.hasClass("ui-rating-cancel") ? a.addClass("ui-rating-cancel-empty").removeClass("ui-rating-cancel-full") : a.prevAll().addBack().not(".ui-rating-cancel").removeClass("ui-rating-hover") }, click: function (a) { var b = c(a.target), f = d.cancelValue; b.parents(".content-box-content:first").removeClass("formerror"); b.hasClass("ui-rating-cancel") ? e.empty(b, b.parent()) : (b.closest(".ui-rating-star").prevAll().addBack().not(".ui-rating-cancel").prop("className", "ui-rating-star ui-rating-full"), b.closest(".ui-rating-star").nextAll().not(".ui-rating-cancel").prop("className", "ui-rating-star ui-rating-empty"), b.siblings(".ui-rating-cancel").prop("className", "ui-rating-cancel ui-rating-cancel-empty"), f = b.val()); a.data.hasChanged || c(a.data.selectBox).val(f).trigger("change") }, change: function (a) { var b = c(this).val(); e.setValue(b, a.data.container, a.data.selectBox) }, setValue: function (a, b, d) { var g = { target: null, data: {} }; g.target = c(".ui-rating-star[value=" + a + "]", b); g.data.selectBox = d; g.data.hasChanged = !0; e.click(g) }, empty: function (a, b) { b.find(".ui-rating-star").removeClass("ui-rating-full"); b.find(".ui-rating-star").addClass("ui-rating-empty"); a.prop("className", "ui-rating-cancel ui-rating-cancel-empty").nextAll().prop("className", "ui-rating-star ui-rating-empty") } }; return this.each(function () { var a = c(this), b, f; "select-one" !== this.type || a.prop("hasProcessed") || (h && c.extend(d, h), a.hide(), a.prop("hasProcessed", !0), b = c("<div/>").prop({ title: this.title, className: "ui-rating" }), a.children("option").each(function () { if ("" != this.value) { var a = c("<a/>"); a.prop({ className: "ui-rating-star ui-rating-empty", title: c(this).text(), value: this.value }).appendTo(b); c(this).is(":selected") && a.prevAll().addBack().removeClass("ui-rating-empty").addClass("ui-rating-full") } }), !0 == d.showCancel && c("<a/>").prop({ className: "ui-rating-cancel ui-rating-cancel-empty", title: d.cancelTitle }).appendTo(b), 0 !== a.children("option:selected").size() ? e.setValue(a.val(), b, a) : (f = null !== d.startValue ? d.startValue : d.cancelValue, e.setValue(f, b, a), a.val(f)), !0 !== d.disabled && !0 !== a.prop("disabled") ? b.bind("mouseover", e.hoverOver).bind("mouseout", e.hoverOut).bind("click", { selectBox: a }, e.click) : b.find("a").css("cursor", "not-allowed"), b.insertAfter(a), a.bind("change", { selectBox: a, container: b }, e.change)) }) } })(jQuery);
/*prettifier_~/js/prettifier/prettify.js~/js/prettifier/lang-vb.js~/js/prettifier/lang-sql.js~/js/prettifier/lang-css.js_key*/
var q=null;window.PR_SHOULD_USE_CONTINUATION=!0;
(function(){function L(a){function m(a){var f=a.charCodeAt(0);if(f!==92)return f;var b=a.charAt(1);return(f=r[b])?f:"0"<=b&&b<="7"?parseInt(a.substring(1),8):b==="u"||b==="x"?parseInt(a.substring(2),16):a.charCodeAt(1)}function e(a){if(a<32)return(a<16?"\\x0":"\\x")+a.toString(16);a=String.fromCharCode(a);if(a==="\\"||a==="-"||a==="["||a==="]")a="\\"+a;return a}function h(a){for(var f=a.substring(1,a.length-1).match(/\\u[\dA-Fa-f]{4}|\\x[\dA-Fa-f]{2}|\\[0-3][0-7]{0,2}|\\[0-7]{1,2}|\\[\S\s]|[^\\]/g),a=
[],b=[],o=f[0]==="^",c=o?1:0,i=f.length;c<i;++c){var j=f[c];if(/\\[bdsw]/i.test(j))a.push(j);else{var j=m(j),d;c+2<i&&"-"===f[c+1]?(d=m(f[c+2]),c+=2):d=j;b.push([j,d]);d<65||j>122||(d<65||j>90||b.push([Math.max(65,j)|32,Math.min(d,90)|32]),d<97||j>122||b.push([Math.max(97,j)&-33,Math.min(d,122)&-33]))}}b.sort(function(a,f){return a[0]-f[0]||f[1]-a[1]});f=[];j=[NaN,NaN];for(c=0;c<b.length;++c)i=b[c],i[0]<=j[1]+1?j[1]=Math.max(j[1],i[1]):f.push(j=i);b=["["];o&&b.push("^");b.push.apply(b,a);for(c=0;c<
f.length;++c)i=f[c],b.push(e(i[0])),i[1]>i[0]&&(i[1]+1>i[0]&&b.push("-"),b.push(e(i[1])));b.push("]");return b.join("")}function y(a){for(var f=a.source.match(/\[(?:[^\\\]]|\\[\S\s])*]|\\u[\dA-Fa-f]{4}|\\x[\dA-Fa-f]{2}|\\\d+|\\[^\dux]|\(\?[!:=]|[()^]|[^()[\\^]+/g),b=f.length,d=[],c=0,i=0;c<b;++c){var j=f[c];j==="("?++i:"\\"===j.charAt(0)&&(j=+j.substring(1))&&j<=i&&(d[j]=-1)}for(c=1;c<d.length;++c)-1===d[c]&&(d[c]=++t);for(i=c=0;c<b;++c)j=f[c],j==="("?(++i,d[i]===void 0&&(f[c]="(?:")):"\\"===j.charAt(0)&&
(j=+j.substring(1))&&j<=i&&(f[c]="\\"+d[i]);for(i=c=0;c<b;++c)"^"===f[c]&&"^"!==f[c+1]&&(f[c]="");if(a.ignoreCase&&s)for(c=0;c<b;++c)j=f[c],a=j.charAt(0),j.length>=2&&a==="["?f[c]=h(j):a!=="\\"&&(f[c]=j.replace(/[A-Za-z]/g,function(a){a=a.charCodeAt(0);return"["+String.fromCharCode(a&-33,a|32)+"]"}));return f.join("")}for(var t=0,s=!1,l=!1,p=0,d=a.length;p<d;++p){var g=a[p];if(g.ignoreCase)l=!0;else if(/[a-z]/i.test(g.source.replace(/\\u[\da-f]{4}|\\x[\da-f]{2}|\\[^UXux]/gi,""))){s=!0;l=!1;break}}for(var r=
{b:8,t:9,n:10,v:11,f:12,r:13},n=[],p=0,d=a.length;p<d;++p){g=a[p];if(g.global||g.multiline)throw Error(""+g);n.push("(?:"+y(g)+")")}return RegExp(n.join("|"),l?"gi":"g")}function M(a){function m(a){switch(a.nodeType){case 1:if(e.test(a.className))break;for(var g=a.firstChild;g;g=g.nextSibling)m(g);g=a.nodeName;if("BR"===g||"LI"===g)h[s]="\n",t[s<<1]=y++,t[s++<<1|1]=a;break;case 3:case 4:g=a.nodeValue,g.length&&(g=p?g.replace(/\r\n?/g,"\n"):g.replace(/[\t\n\r ]+/g," "),h[s]=g,t[s<<1]=y,y+=g.length,
t[s++<<1|1]=a)}}var e=/(?:^|\s)nocode(?:\s|$)/,h=[],y=0,t=[],s=0,l;a.currentStyle?l=a.currentStyle.whiteSpace:window.getComputedStyle&&(l=document.defaultView.getComputedStyle(a,q).getPropertyValue("white-space"));var p=l&&"pre"===l.substring(0,3);m(a);return{a:h.join("").replace(/\n$/,""),c:t}}function B(a,m,e,h){m&&(a={a:m,d:a},e(a),h.push.apply(h,a.e))}function x(a,m){function e(a){for(var l=a.d,p=[l,"pln"],d=0,g=a.a.match(y)||[],r={},n=0,z=g.length;n<z;++n){var f=g[n],b=r[f],o=void 0,c;if(typeof b===
"string")c=!1;else{var i=h[f.charAt(0)];if(i)o=f.match(i[1]),b=i[0];else{for(c=0;c<t;++c)if(i=m[c],o=f.match(i[1])){b=i[0];break}o||(b="pln")}if((c=b.length>=5&&"lang-"===b.substring(0,5))&&!(o&&typeof o[1]==="string"))c=!1,b="src";c||(r[f]=b)}i=d;d+=f.length;if(c){c=o[1];var j=f.indexOf(c),k=j+c.length;o[2]&&(k=f.length-o[2].length,j=k-c.length);b=b.substring(5);B(l+i,f.substring(0,j),e,p);B(l+i+j,c,C(b,c),p);B(l+i+k,f.substring(k),e,p)}else p.push(l+i,b)}a.e=p}var h={},y;(function(){for(var e=a.concat(m),
l=[],p={},d=0,g=e.length;d<g;++d){var r=e[d],n=r[3];if(n)for(var k=n.length;--k>=0;)h[n.charAt(k)]=r;r=r[1];n=""+r;p.hasOwnProperty(n)||(l.push(r),p[n]=q)}l.push(/[\S\s]/);y=L(l)})();var t=m.length;return e}function u(a){var m=[],e=[];a.tripleQuotedStrings?m.push(["str",/^(?:'''(?:[^'\\]|\\[\S\s]|''?(?=[^']))*(?:'''|$)|"""(?:[^"\\]|\\[\S\s]|""?(?=[^"]))*(?:"""|$)|'(?:[^'\\]|\\[\S\s])*(?:'|$)|"(?:[^"\\]|\\[\S\s])*(?:"|$))/,q,"'\""]):a.multiLineStrings?m.push(["str",/^(?:'(?:[^'\\]|\\[\S\s])*(?:'|$)|"(?:[^"\\]|\\[\S\s])*(?:"|$)|`(?:[^\\`]|\\[\S\s])*(?:`|$))/,
q,"'\"`"]):m.push(["str",/^(?:'(?:[^\n\r'\\]|\\.)*(?:'|$)|"(?:[^\n\r"\\]|\\.)*(?:"|$))/,q,"\"'"]);a.verbatimStrings&&e.push(["str",/^@"(?:[^"]|"")*(?:"|$)/,q]);var h=a.hashComments;h&&(a.cStyleComments?(h>1?m.push(["com",/^#(?:##(?:[^#]|#(?!##))*(?:###|$)|.*)/,q,"#"]):m.push(["com",/^#(?:(?:define|elif|else|endif|error|ifdef|include|ifndef|line|pragma|undef|warning)\b|[^\n\r]*)/,q,"#"]),e.push(["str",/^<(?:(?:(?:\.\.\/)*|\/?)(?:[\w-]+(?:\/[\w-]+)+)?[\w-]+\.h|[a-z]\w*)>/,q])):m.push(["com",/^#[^\n\r]*/,
q,"#"]));a.cStyleComments&&(e.push(["com",/^\/\/[^\n\r]*/,q]),e.push(["com",/^\/\*[\S\s]*?(?:\*\/|$)/,q]));a.regexLiterals&&e.push(["lang-regex",/^(?:^^\.?|[!+-]|!=|!==|#|%|%=|&|&&|&&=|&=|\(|\*|\*=|\+=|,|-=|->|\/|\/=|:|::|;|<|<<|<<=|<=|=|==|===|>|>=|>>|>>=|>>>|>>>=|[?@[^]|\^=|\^\^|\^\^=|{|\||\|=|\|\||\|\|=|~|break|case|continue|delete|do|else|finally|instanceof|return|throw|try|typeof)\s*(\/(?=[^*/])(?:[^/[\\]|\\[\S\s]|\[(?:[^\\\]]|\\[\S\s])*(?:]|$))+\/)/]);(h=a.types)&&e.push(["typ",h]);a=(""+a.keywords).replace(/^ | $/g,
"");a.length&&e.push(["kwd",RegExp("^(?:"+a.replace(/[\s,]+/g,"|")+")\\b"),q]);m.push(["pln",/^\s+/,q," \r\n\t\xa0"]);e.push(["lit",/^@[$_a-z][\w$@]*/i,q],["typ",/^(?:[@_]?[A-Z]+[a-z][\w$@]*|\w+_t\b)/,q],["pln",/^[$_a-z][\w$@]*/i,q],["lit",/^(?:0x[\da-f]+|(?:\d(?:_\d+)*\d*(?:\.\d*)?|\.\d\+)(?:e[+-]?\d+)?)[a-z]*/i,q,"0123456789"],["pln",/^\\[\S\s]?/,q],["pun",/^.[^\s\w"-$'./@\\`]*/,q]);return x(m,e)}function D(a,m){function e(a){switch(a.nodeType){case 1:if(k.test(a.className))break;if("BR"===a.nodeName)h(a),
a.parentNode&&a.parentNode.removeChild(a);else for(a=a.firstChild;a;a=a.nextSibling)e(a);break;case 3:case 4:if(p){var b=a.nodeValue,d=b.match(t);if(d){var c=b.substring(0,d.index);a.nodeValue=c;(b=b.substring(d.index+d[0].length))&&a.parentNode.insertBefore(s.createTextNode(b),a.nextSibling);h(a);c||a.parentNode.removeChild(a)}}}}function h(a){function b(a,d){var e=d?a.cloneNode(!1):a,f=a.parentNode;if(f){var f=b(f,1),g=a.nextSibling;f.appendChild(e);for(var h=g;h;h=g)g=h.nextSibling,f.appendChild(h)}return e}
for(;!a.nextSibling;)if(a=a.parentNode,!a)return;for(var a=b(a.nextSibling,0),e;(e=a.parentNode)&&e.nodeType===1;)a=e;d.push(a)}var k=/(?:^|\s)nocode(?:\s|$)/,t=/\r\n?|\n/,s=a.ownerDocument,l;a.currentStyle?l=a.currentStyle.whiteSpace:window.getComputedStyle&&(l=s.defaultView.getComputedStyle(a,q).getPropertyValue("white-space"));var p=l&&"pre"===l.substring(0,3);for(l=s.createElement("LI");a.firstChild;)l.appendChild(a.firstChild);for(var d=[l],g=0;g<d.length;++g)e(d[g]);m===(m|0)&&d[0].setAttribute("value",
m);var r=s.createElement("OL");r.className="linenums";for(var n=Math.max(0,m-1|0)||0,g=0,z=d.length;g<z;++g)l=d[g],l.className="L"+(g+n)%10,l.firstChild||l.appendChild(s.createTextNode("\xa0")),r.appendChild(l);a.appendChild(r)}function k(a,m){for(var e=m.length;--e>=0;){var h=m[e];A.hasOwnProperty(h)?window.console&&console.warn("cannot override language handler %s",h):A[h]=a}}function C(a,m){if(!a||!A.hasOwnProperty(a))a=/^\s*</.test(m)?"default-markup":"default-code";return A[a]}function E(a){var m=
a.g;try{var e=M(a.h),h=e.a;a.a=h;a.c=e.c;a.d=0;C(m,h)(a);var k=/\bMSIE\b/.test(navigator.userAgent),m=/\n/g,t=a.a,s=t.length,e=0,l=a.c,p=l.length,h=0,d=a.e,g=d.length,a=0;d[g]=s;var r,n;for(n=r=0;n<g;)d[n]!==d[n+2]?(d[r++]=d[n++],d[r++]=d[n++]):n+=2;g=r;for(n=r=0;n<g;){for(var z=d[n],f=d[n+1],b=n+2;b+2<=g&&d[b+1]===f;)b+=2;d[r++]=z;d[r++]=f;n=b}for(d.length=r;h<p;){var o=l[h+2]||s,c=d[a+2]||s,b=Math.min(o,c),i=l[h+1],j;if(i.nodeType!==1&&(j=t.substring(e,b))){k&&(j=j.replace(m,"\r"));i.nodeValue=
j;var u=i.ownerDocument,v=u.createElement("SPAN");v.className=d[a+1];var x=i.parentNode;x.replaceChild(v,i);v.appendChild(i);e<o&&(l[h+1]=i=u.createTextNode(t.substring(b,o)),x.insertBefore(i,v.nextSibling))}e=b;e>=o&&(h+=2);e>=c&&(a+=2)}}catch(w){"console"in window&&console.log(w&&w.stack?w.stack:w)}}var v=["break,continue,do,else,for,if,return,while"],w=[[v,"auto,case,char,const,default,double,enum,extern,float,goto,int,long,register,short,signed,sizeof,static,struct,switch,typedef,union,unsigned,void,volatile"],
"catch,class,delete,false,import,new,operator,private,protected,public,this,throw,true,try,typeof"],F=[w,"alignof,align_union,asm,axiom,bool,concept,concept_map,const_cast,constexpr,decltype,dynamic_cast,explicit,export,friend,inline,late_check,mutable,namespace,nullptr,reinterpret_cast,static_assert,static_cast,template,typeid,typename,using,virtual,where"],G=[w,"abstract,boolean,byte,extends,final,finally,implements,import,instanceof,null,native,package,strictfp,super,synchronized,throws,transient"],
H=[G,"as,base,by,checked,decimal,delegate,descending,dynamic,event,fixed,foreach,from,group,implicit,in,interface,internal,into,is,lock,object,out,override,orderby,params,partial,readonly,ref,sbyte,sealed,stackalloc,string,select,uint,ulong,unchecked,unsafe,ushort,var"],w=[w,"debugger,eval,export,function,get,null,set,undefined,var,with,Infinity,NaN"],I=[v,"and,as,assert,class,def,del,elif,except,exec,finally,from,global,import,in,is,lambda,nonlocal,not,or,pass,print,raise,try,with,yield,False,True,None"],
J=[v,"alias,and,begin,case,class,def,defined,elsif,end,ensure,false,in,module,next,nil,not,or,redo,rescue,retry,self,super,then,true,undef,unless,until,when,yield,BEGIN,END"],v=[v,"case,done,elif,esac,eval,fi,function,in,local,set,then,until"],K=/^(DIR|FILE|vector|(de|priority_)?queue|list|stack|(const_)?iterator|(multi)?(set|map)|bitset|u?(int|float)\d*)/,N=/\S/,O=u({keywords:[F,H,w,"caller,delete,die,do,dump,elsif,eval,exit,foreach,for,goto,if,import,last,local,my,next,no,our,print,package,redo,require,sub,undef,unless,until,use,wantarray,while,BEGIN,END"+
I,J,v],hashComments:!0,cStyleComments:!0,multiLineStrings:!0,regexLiterals:!0}),A={};k(O,["default-code"]);k(x([],[["pln",/^[^<?]+/],["dec",/^<!\w[^>]*(?:>|$)/],["com",/^<\!--[\S\s]*?(?:--\>|$)/],["lang-",/^<\?([\S\s]+?)(?:\?>|$)/],["lang-",/^<%([\S\s]+?)(?:%>|$)/],["pun",/^(?:<[%?]|[%?]>)/],["lang-",/^<xmp\b[^>]*>([\S\s]+?)<\/xmp\b[^>]*>/i],["lang-js",/^<script\b[^>]*>([\S\s]*?)(<\/script\b[^>]*>)/i],["lang-css",/^<style\b[^>]*>([\S\s]*?)(<\/style\b[^>]*>)/i],["lang-in.tag",/^(<\/?[a-z][^<>]*>)/i]]),
["default-markup","htm","html","mxml","xhtml","xml","xsl"]);k(x([["pln",/^\s+/,q," \t\r\n"],["atv",/^(?:"[^"]*"?|'[^']*'?)/,q,"\"'"]],[["tag",/^^<\/?[a-z](?:[\w-.:]*\w)?|\/?>$/i],["atn",/^(?!style[\s=]|on)[a-z](?:[\w:-]*\w)?/i],["lang-uq.val",/^=\s*([^\s"'>]*(?:[^\s"'/>]|\/(?=\s)))/],["pun",/^[/<->]+/],["lang-js",/^on\w+\s*=\s*"([^"]+)"/i],["lang-js",/^on\w+\s*=\s*'([^']+)'/i],["lang-js",/^on\w+\s*=\s*([^\s"'>]+)/i],["lang-css",/^style\s*=\s*"([^"]+)"/i],["lang-css",/^style\s*=\s*'([^']+)'/i],["lang-css",
/^style\s*=\s*([^\s"'>]+)/i]]),["in.tag"]);k(x([],[["atv",/^[\S\s]+/]]),["uq.val"]);k(u({keywords:F,hashComments:!0,cStyleComments:!0,types:K}),["c","cc","cpp","cxx","cyc","m"]);k(u({keywords:"null,true,false"}),["json"]);k(u({keywords:H,hashComments:!0,cStyleComments:!0,verbatimStrings:!0,types:K}),["cs"]);k(u({keywords:G,cStyleComments:!0}),["java"]);k(u({keywords:v,hashComments:!0,multiLineStrings:!0}),["bsh","csh","sh"]);k(u({keywords:I,hashComments:!0,multiLineStrings:!0,tripleQuotedStrings:!0}),
["cv","py"]);k(u({keywords:"caller,delete,die,do,dump,elsif,eval,exit,foreach,for,goto,if,import,last,local,my,next,no,our,print,package,redo,require,sub,undef,unless,until,use,wantarray,while,BEGIN,END",hashComments:!0,multiLineStrings:!0,regexLiterals:!0}),["perl","pl","pm"]);k(u({keywords:J,hashComments:!0,multiLineStrings:!0,regexLiterals:!0}),["rb"]);k(u({keywords:w,cStyleComments:!0,regexLiterals:!0}),["js"]);k(u({keywords:"all,and,by,catch,class,else,extends,false,finally,for,if,in,is,isnt,loop,new,no,not,null,of,off,on,or,return,super,then,true,try,unless,until,when,while,yes",
hashComments:3,cStyleComments:!0,multilineStrings:!0,tripleQuotedStrings:!0,regexLiterals:!0}),["coffee"]);k(x([],[["str",/^[\S\s]+/]]),["regex"]);window.prettyPrintOne=function(a,m,e){var h=document.createElement("PRE");h.innerHTML=a;e&&D(h,e);E({g:m,i:e,h:h});return h.innerHTML};window.prettyPrint=function(a){function m(){for(var e=window.PR_SHOULD_USE_CONTINUATION?l.now()+250:Infinity;p<h.length&&l.now()<e;p++){var n=h[p],k=n.className;if(k.indexOf("prettyprint")>=0){var k=k.match(g),f,b;if(b=
!k){b=n;for(var o=void 0,c=b.firstChild;c;c=c.nextSibling)var i=c.nodeType,o=i===1?o?b:c:i===3?N.test(c.nodeValue)?b:o:o;b=(f=o===b?void 0:o)&&"CODE"===f.tagName}b&&(k=f.className.match(g));k&&(k=k[1]);b=!1;for(o=n.parentNode;o;o=o.parentNode)if((o.tagName==="pre"||o.tagName==="code"||o.tagName==="xmp")&&o.className&&o.className.indexOf("prettyprint")>=0){b=!0;break}b||((b=(b=n.className.match(/\blinenums\b(?::(\d+))?/))?b[1]&&b[1].length?+b[1]:!0:!1)&&D(n,b),d={g:k,h:n,i:b},E(d))}}p<h.length?setTimeout(m,
250):a&&a()}for(var e=[document.getElementsByTagName("pre"),document.getElementsByTagName("code"),document.getElementsByTagName("xmp")],h=[],k=0;k<e.length;++k)for(var t=0,s=e[k].length;t<s;++t)h.push(e[k][t]);var e=q,l=Date;l.now||(l={now:function(){return+new Date}});var p=0,d,g=/\blang(?:uage)?-([\w.]+)(?!\S)/;m()};window.PR={createSimpleLexer:x,registerLangHandler:k,sourceDecorator:u,PR_ATTRIB_NAME:"atn",PR_ATTRIB_VALUE:"atv",PR_COMMENT:"com",PR_DECLARATION:"dec",PR_KEYWORD:"kwd",PR_LITERAL:"lit",
PR_NOCODE:"nocode",PR_PLAIN:"pln",PR_PUNCTUATION:"pun",PR_SOURCE:"src",PR_STRING:"str",PR_TAG:"tag",PR_TYPE:"typ"}})();

PR.registerLangHandler(PR.createSimpleLexer([[PR.PR_PLAIN,/^[\t\n\r \xA0\u2028\u2029]+/,null,"\t\n\r \u00a0\u2028\u2029"],[PR.PR_STRING,/^(?:[\"\u201C\u201D](?:[^\"\u201C\u201D]|[\"\u201C\u201D]{2})(?:[\"\u201C\u201D]c|$)|[\"\u201C\u201D](?:[^\"\u201C\u201D]|[\"\u201C\u201D]{2})*(?:[\"\u201C\u201D]|$))/i,null,'"\u201c\u201d'],[PR.PR_COMMENT,/^[\'\u2018\u2019][^\r\n\u2028\u2029]*/,null,"'\u2018\u2019"]],[[PR.PR_KEYWORD,/^(?:AddHandler|AddressOf|Alias|And|AndAlso|Ansi|As|Assembly|Auto|Boolean|ByRef|Byte|ByVal|Call|Case|Catch|CBool|CByte|CChar|CDate|CDbl|CDec|Char|CInt|Class|CLng|CObj|Const|CShort|CSng|CStr|CType|Date|Decimal|Declare|Default|Delegate|Dim|DirectCast|Do|Double|Each|Else|ElseIf|End|EndIf|Enum|Erase|Error|Event|Exit|Finally|For|Friend|Function|Get|GetType|GoSub|GoTo|Handles|If|Implements|Imports|In|Inherits|Integer|Interface|Is|Let|Lib|Like|Long|Loop|Me|Mod|Module|MustInherit|MustOverride|MyBase|MyClass|Namespace|New|Next|Not|NotInheritable|NotOverridable|Object|On|Option|Optional|Or|OrElse|Overloads|Overridable|Overrides|ParamArray|Preserve|Private|Property|Protected|Public|RaiseEvent|ReadOnly|ReDim|RemoveHandler|Resume|Return|Select|Set|Shadows|Shared|Short|Single|Static|Step|Stop|String|Structure|Sub|SyncLock|Then|Throw|To|Try|TypeOf|Unicode|Until|Variant|Wend|When|While|With|WithEvents|WriteOnly|Xor|EndIf|GoSub|Let|Variant|Wend)\b/i,
null],[PR.PR_COMMENT,/^REM[^\r\n\u2028\u2029]*/i],[PR.PR_LITERAL,/^(?:True\b|False\b|Nothing\b|\d+(?:E[+\-]?\d+[FRD]?|[FRDSIL])?|(?:&H[0-9A-F]+|&O[0-7]+)[SIL]?|\d*\.\d+(?:E[+\-]?\d+)?[FRD]?|#\s+(?:\d+[\-\/]\d+[\-\/]\d+(?:\s+\d+:\d+(?::\d+)?(\s*(?:AM|PM))?)?|\d+:\d+(?::\d+)?(\s*(?:AM|PM))?)\s+#)/i],[PR.PR_PLAIN,/^(?:(?:[a-z]|_\w)\w*|\[(?:[a-z]|_\w)\w*\])/i],[PR.PR_PUNCTUATION,/^[^\w\t\n\r \"\'\[\]\xA0\u2018\u2019\u201C\u201D\u2028\u2029]+/],[PR.PR_PUNCTUATION,/^(?:\[|\])/]]),["vb","vbs"]);
PR.registerLangHandler(PR.createSimpleLexer([["pln",/^[\t\n\r \xa0]+/,null,"\t\n\r �\xa0"],["str",/^(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,null,"\"'"]],[["com",/^(?:--[^\n\r]*|\/\*[\S\s]*?(?:\*\/|$))/],["kwd",/^(?:add|all|alter|and|any|as|asc|authorization|backup|begin|between|break|browse|bulk|by|cascade|case|check|checkpoint|close|clustered|coalesce|collate|column|commit|compute|constraint|contains|containstable|continue|convert|create|cross|current|current_date|current_time|current_timestamp|current_user|cursor|database|dbcc|deallocate|declare|default|delete|deny|desc|disk|distinct|distributed|double|drop|dummy|dump|else|end|errlvl|escape|except|exec|execute|exists|exit|fetch|file|fillfactor|for|foreign|freetext|freetexttable|from|full|function|goto|grant|group|having|holdlock|identity|identitycol|identity_insert|if|in|index|inner|insert|intersect|into|is|join|key|kill|left|like|lineno|load|match|merge|national|nocheck|nonclustered|not|null|nullif|of|off|offsets|on|open|opendatasource|openquery|openrowset|openxml|option|or|order|outer|over|percent|plan|precision|primary|print|proc|procedure|public|raiserror|read|readtext|reconfigure|references|replication|restore|restrict|return|revoke|right|rollback|rowcount|rowguidcol|rule|save|schema|select|session_user|set|setuser|shutdown|some|statistics|system_user|table|textsize|then|to|top|tran|transaction|trigger|truncate|tsequal|union|unique|update|updatetext|use|user|using|values|varying|view|waitfor|when|where|while|with|writetext)(?=[^\w-]|$)/i,
null],["lit",/^[+-]?(?:0x[\da-f]+|(?:\.\d+|\d+(?:\.\d*)?)(?:e[+-]?\d+)?)/i],["pln",/^[_a-z][\w-]*/i],["pun",/^[^\w\t\n\r "'\xa0][^\w\t\n\r "'+\xa0-]*/]]),["sql"]);

PR.registerLangHandler(PR.createSimpleLexer([["pln",/^[\t\n\f\r ]+/,null," \t\r\n"]],[["str",/^"(?:[^\n\f\r"\\]|\\(?:\r\n?|\n|\f)|\\[\S\s])*"/,null],["str",/^'(?:[^\n\f\r'\\]|\\(?:\r\n?|\n|\f)|\\[\S\s])*'/,null],["lang-css-str",/^url\(([^"')]*)\)/i],["kwd",/^(?:url|rgb|!important|@import|@page|@media|@charset|inherit)(?=[^\w-]|$)/i,null],["lang-css-kw",/^(-?(?:[_a-z]|\\[\da-f]+ ?)(?:[\w-]|\\\\[\da-f]+ ?)*)\s*:/i],["com",/^\/\*[^*]*\*+(?:[^*/][^*]*\*+)*\//],["com",
/^(?:<\!--|--\>)/],["lit",/^(?:\d+|\d*\.\d+)(?:%|[a-z]+)?/i],["lit",/^#[\da-f]{3,6}/i],["pln",/^-?(?:[_a-z]|\\[\da-f]+ ?)(?:[\w-]|\\\\[\da-f]+ ?)*/i],["pun",/^[^\s\w"']+/]]),["css"]);PR.registerLangHandler(PR.createSimpleLexer([],[["kwd",/^-?(?:[_a-z]|\\[\da-f]+ ?)(?:[\w-]|\\\\[\da-f]+ ?)*/i]]),["css-kw"]);PR.registerLangHandler(PR.createSimpleLexer([],[["str",/^[^"')]+/]]),["css-str"]);

/*regularmaster_key*/

function RegisterBackToTopScroller() {

    if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;
 
    var $backtotop = $('div.backtotop');
    var $main = $('#main');

    $backtotop.click(function () {
        $.scrollTo(0, { duration: 200 });
        return false;
    });

    var firstScrolled = false;

    $win.bind('scrollstop', function () {

        if ($win.scrollTop() > 500) {
            if (!firstScrolled) {

                var marginLeft = (($win.width() - $main.width()) / 2) > 35 ? 10 : -45;

                $backtotop.css('margin-left', $main.width() + marginLeft);
                firstScrolled = true;
            }
            $backtotop.fadeIn('fast');
        } else {
            $backtotop.fadeOut('fast');
        }
    });
}

function RegisterUniversalMessageMemberHover() {
    
    if (cMemberInfo.usingMobileTheme || cMemberInfo.isMobileDevice) return;

    $body
        .on('mouseenter', 'a.messagelink', function () {

            var $lnk = $(this);
            var options = {
                overwrite: false,
                position: { at: 'middle left', my: 'bottom left', adjust: { x: 0, y: -18 }, viewport: $win },
                style: { tip: false },
                show: {
                    delay: 500,
                    effect: function () {
                        $(this).fadeIn(150);
                    }
                },
                hide: { delay: 100, fixed: false, inactive: 5000, event: 'click mouseleave' },
                events: {
                    render: function () {
                        $(this).css({ 'max-width': '450px', 'line-height': 1.6 });
                    },
                    hide: null
                }
            };

            var title = $lnk.attr('title');
            if (title)
                qtip.notice($lnk, title.replace(/\</gi, '&lt;').replace(/\>/gi, '&gt;'), options);

            return false;
        })
        .on('mouseover', 'a.authorlink', function () {
            var $lnk = $(this);
            var memID = $lnk.data('memid');

            if (memID < 0) {
                $lnk.attr('href', 'javascript:void(0)');
                return false;   
            }

            var adjustx = $lnk.data('adjustx');

            var options = {
                position: { at: 'right middle', my: 'left middle', adjust: { x: (adjustx ? adjustx : 5) }, viewport: $win },
                show: { delay: 500 },
                hide: { delay: 200, event: 'click mouseleave', fixed: true },
                style: { tip: false }
            };

            var pmlinktmpl = '{0}pmsend.aspx?toMemId={1}';

            if (cMemberInfo.popupPosting)
                pmlinktmpl = 'javascript:void(popRadWin(\'' + pmlinktmpl + '&pop=true\'))';

            var linkContent = String.format(
                '<a href="{0}Posts/{1}" class="{2}">{4}</a>{3}' +
                (cFeatureInfo.galleryActive ? '<a href="{0}photosearch.aspx?albummemid={1}" class="{2}">{5}</a>{3}' : '') +
                (cMemberInfo.memID != memID && cMemberInfo.memID != -1 && cMemberInfo.pmAllowed ?
                    '<a href="{0}pm.aspx?senderID={1}" class="{2}">{6}</a>{3}' +
                    '<a href="' + pmlinktmpl + '" class="{2}">{7}</a>' : ''),
            cPathInfo.ForumDir, memID, 'medium postButtonDropdown', '<div class="divider"></div>',
            ln.profRecentPosts, ln.galListAlbumsByMember, ln.pmListPMsByAuthor, ln.tmMenuSendPM);

            qtip.notice($lnk, linkContent, options);

            // if ($lnk.hasClass('pmMemberLink')) else if ($lnk.hasClass('photoMemberLink'))

        });
}

$doc.ready(function () {
    RegisterBackToTopScroller();
    RegisterUniversalMessageMemberHover();
});
/*topNavigation_key*/

function registerSearchBoxes() {

    $('#masterSearchButton').click(function () {
        performTopSearch($('#masterphrase'));
        return false;
    });

    $('#masterphrase').keypress(function (e) {
        
        if (isEnterKeyPressed(e)) {
            performTopSearch($(this));
            return false;
        }
        return true;
    });

}

function performTopSearch($theTextbox) {

    var phrase = $theTextbox.val().trim();

    if (phrase.length < 3) {
        showNoticeToFilterSearchbox($theTextbox, ln.srTermTooShort);
        return false;
    }

    showNoticeToFilterSearchbox($theTextbox, ln.srSearchWaitingMsg);

    initiateSearchSimilarThreads(phrase, 'ALL', function (r) {
        var result = r.d; //[asyncID, server now, highlight] or [Integer delay, "" ,""]
        var aid = result[0];

        if (!isNaN(aid)) {
            showNoticeToFilterSearchbox($theTextbox, ln.srSearchWaitAbit);
            return false;
        }

        if (aid == 'reentersearch') {
            showNoticeToFilterSearchbox($theTextbox, ln.srSearchNoResultMsg);
            return false;
        } else { // start tracking
            trackSearchByAid(aid, result[1], $theTextbox);
        }

        return true;
    }, 200, ['Normal', 'Exact']);

    return true;
}

function trackSearchByAid(aid, beginTime, $theTextbox) {
    window.dropdownfilterInterval = setInterval(function () {
        searchCheckIfComplete(aid, beginTime, function (result) {
            var searchid = result.d;
            switch (searchid) {
                case 0: // not done, keep checking at interval
                    break;
                case -1: // no result
                    clearInterval(window.dropdownfilterInterval);
                    showNoticeToFilterSearchbox($theTextbox, ln.srSearchNoResultMsg);
                    break;
                default:
                    clearInterval(window.dropdownfilterInterval);
                    $theTextbox.qtip('hide');
                    $theTextbox.blur();

                    loadSearchResultOnMasterPhrase($theTextbox, searchid);

                    break;
            }
        });
    }, 1000);
}

function loadSearchResultOnMasterPhrase($theTextbox, searchid) {
    var options = $.extend(true, getTopNavDropDownTipOptions(), {
        style: { width: 'auto' },
        events: {
            render: function (e, api) {
                var $thistip = $(this);
                var $content = $thistip.children(qtip.contentSelector);
                $content.css({ 'overflow': 'auto' })
                    .load(cPathInfo.ForumDir + 'ws/TopNav.aspx?s=' + searchid.toString() + ' #navdata', function () {
                        api.set('position.my', 'top right');
                        api.set('position.at', 'bottom right');

                        $content.append(String.format(
                            '<div class="center bMargin10 medium"><a href="{0}search?searchid={1}&phrase={2}&high={2}">{3} <i class="icon-forward"></i></a></div>',
                            cPathInfo.ForumDir, searchid, encodeURIComponent($theTextbox.val()), ln.viewMoreDesc));
                    });

            },
            visible: function () {
                var $thistip = $(this);
                var $content = $thistip.children(qtip.contentSelector);
                var maxHeight = getTopNavDropDownContentMaxHeight();
                $content.css({ 'max-height': maxHeight });
            }
        }
    });

    $theTextbox.qtip(options);
}

function showNoticeToFilterSearchbox($theTextbox, content) {
    if ($theTextbox.data('notip') === true) return;
    qtip.notice($theTextbox, content, { position: { adjust: { x: -10}} });
}

/*function sendMasterSearchString() {
    var masterPhrase = $('#masterphrase').val().trim();
    if (masterPhrase == '') return;
    
    self.location.href = cPathInfo.ForumDir + 'search.aspx?phrase=' +
        escape(masterPhrase.replace(/\</gi, ''));
} */
function registerLoginbox($theLink, redirectTo) {

    if (redirectTo != '') {

        $theLink.click(function () {
            self.location.href = redirectTo +
                    (redirectTo.indexOf('?') >= 0 ? '&' : '?') +
                        'ReturnUrl=' + escape(cPathInfo.Url);
        });
        return;
    }

    if (cMemberInfo.isMobileDevice) return; // don't use popup for mobile devices

    var $loginTbl = $('#subnav-loginbox-div');
    var html = $loginTbl.html();

    $loginTbl.remove();

    $theLink.click(function (e, showtarget, additionalOptions) {
        /*        
        consoleLog(showtarget);
        consoleLog((showtarget ? $(showtarget) : $theLink));
        */
        var oriQtipOptions = {
            content: { text: html, title: { text: ln.logTitleDesc, button: true} },
            position: { my: 'top center', at: 'top center',
                adjust: { y: (cMemberInfo.isMobileDevice ? 10 : 190) }, // mobile keyboard will hide input
                target: $win
            },
            show: { event: 'click', ready: true, modal: false },
            hide: { fixed: true, delay: 1000, event: 'unfocus' },
            style: { classes: 'qtip-apglogin qtip-wideshadow ', tip: false, width:620 },
            events: {
                visible: function (e, api) {
                    var $thisTip = $(this);
                    $thisTip.find('input:first').focus();
                    blankRefreshCaptchaForLogin();
                },
                render: function (e, api) {
                    var $thisTip = $(this);

                    $thisTip.find('.fbLogin, .twitterLogin').attr('href', function (ind, attr) {
                        return attr + '&ReturnUrl=' + escape(cPathInfo.Url.replace(/^.*\/\/[^\/]+/, ''));
                    });

                    $body
                    .delegate('.subnav-loginbox input', 'keydown', function (ev) {
                        var code = (ev.keyCode ? ev.keyCode : ev.which);
                        if (code == 13) {
                            $("#subnav-loginbox-submit").click();
                        }
                    })
                    .delegate('#subnav-loginbox-submit', 'click', function () {
                        var ajaxData = {
                            username: $('#subnav-loginbox-u').val().trim(),
                            password: $('#subnav-loginbox-p').val().trim(),
                            captchaVal: $('#subnav-loginbox-c').val().trim(),
                            remember: $('#subnav-loginbox-r').prop('checked')
                        };
                        JQCallWebService('ws/Login.aspx/SignIn', ajaxData,
                            function (r) {
                                var returnedVal = r.d;

                                if (returnedVal.success) {
                                    var selfhref = self.location.href;

                                    if (selfhref.indexOf('/login') > 0 ||
                                        selfhref.indexOf('/social') > 0 ||
                                        selfhref.indexOf('/register') > 0 ||
                                        selfhref.indexOf('/confirmation') > 0) { //prevent reloading of wrong page

                                        self.location.replace(cPathInfo.ForumDir);
                                    } else {
                                        self.location.replace(location.href.split("#")[0]);
                                    }
                                } else {
                                    //consoleLog(returnedVal.resultCode);
                                    switch (returnedVal.resultCode) {
                                        // only display error on captcha, as we need a way to show the various "forget features"            
                                        case -6:
                                            blankRefreshCaptchaForLogin();
                                            qtip.notice($('#subnav-loginbox-c'), ln.formVerificationFailureMsg);
                                            break;
                                        default: // all errors
                                            self.location.href = String.format('{0}login?err={1}&username={2}',
                                                cPathInfo.ForumDir, returnedVal.resultCode, ajaxData.username);
                                            break;
                                    }

                                }
                            }, null, this);

                        return false;
                    });
                }
            }
        };

        $theLink.qtip($.extend(true, {}, oriQtipOptions, additionalOptions));

        return false;
    });

}

function blankRefreshCaptchaForLogin() {
    $('#subnav-loginbox-c').val('');
    $('#subnav-loginbox-captcha:visible')
            .attr('src', pageThemeImageURL + 'CaptchaImage.axd?p=' + randomString(5));
}

function registerGplusButtonClick() {
    $body.on('click', 'a.googleLogin', function () {
        $('#form1')
            .attr('action', cPathInfo.ForumDir + 'sociallogin.aspx?ReturnUrl=' + escape(cPathInfo.Url.replace(/^.*\/\/[^\/]+/, '')))
            .append('<input type="hidden" name="openid_identifier" value="https://www.google.com/accounts/o8/id">')
            .submit();
        return false;
    });
}

    var _paneClass = 'individualSection';
    var _topNavLiDistance = 0;
    
    function getTopNavDropDownContentMaxHeight() {
        if (cMemberInfo.isMobileDevice)
            _topNavLiDistance = 1E5;
        
        if (_topNavLiDistance == 0) {
            var $li = $('ul.topnavTabList').find('li:first');
            _topNavLiDistance = $li.offset().top + $li.innerHeight();
        }

        return $win.height() - _topNavLiDistance - 30;
    }

    function getTopNavDropDownTipOptions() {
        return {
            overwrite: false,
            content: { text: ln.loadingDesc + '.' }, //extra dot to make sure if ln is not available, tip still renders
            position: { my: 'top left', at: 'bottom left'},
            style: { classes: ' qtip-apgmenu ' + (Modernizr.borderimage ? ' qtip-wideshadow ' : ' qtip-iemenushadow '), //Modernizr.borderimage as a test for IE all versions
                        widget: false, tip: false },
            show: { ready: true, event: _mobileAwareEventName, modal: { on: false }, effect: false },
            hide: { fixed: true, delay: 300, event: ((typeof cMemberInfo != 'undefined' && !cMemberInfo.isMobileDevice) ? 'unfocus' : 'mouseleave'), effect: false },
            events : {
                hide: function (e, api) { api.elements.target.removeClass('selected'); }
            }
        };
    }

    function registerTopNavDropDowns() {

        var topTabLiSel = 'ul.topnavTabList > li'; // each top menu button

        // add a down triangle
        var $topLis = $(topTabLiSel);
        $topLis.append('<span class="ui-icon ui-icon-triangle-1-s"></span>');

        $('div.topnavTabs').on(_mobileAwareEventName, topTabLiSel, function () {

            var topTabLiSelClass = 'selected';
            var $thisLi = $(this);

            if ($thisLi.hasClass(topTabLiSelClass)) { // click again close
                $thisLi.removeClass(topTabLiSelClass);
                $thisLi.qtip('hide');
                return true;
            }

            // deselect all other buttons
            $thisLi
                .siblings('li').removeClass(topTabLiSelClass)
                .end().addClass(topTabLiSelClass);

            if (typeof $thisLi.attr('id') != 'undefined') return true;

            var options = $.extend(true, getTopNavDropDownTipOptions(), {
                position: { target: $thisLi.parent().find('li:first') },
                events: {
                    render: function (e, api) {
                        var $thistip = $(this);

                        api.set('content.text', GetHiddenTopMenuContent($thisLi));

                        var filterType = getFilterTypeFromLi($thisLi);

                        if (filterType == 'pm') {
                            api.set('position.target', $thisLi);
                        }

                        $thistip.data('fortype', filterType);

                        registerEventsForTopNavContent($thistip);

                        //$thistip

                    },
                    visible: function (e, api) {
                        var $thisTip = $(this);

                        var $content = $thisTip.children(qtip.contentSelector);

                        $content.css('max-height', getTopNavDropDownContentMaxHeight());

                        var filterType = $thisTip.data('fortype');

                        if (filterType == 'pm') {
                            $thisTip.css('width', 500);
                        } else {
                            $thisTip.css('width', $('#main').width());
                        }

                        setMenuDivsWidth($thisTip);

                        var $prevShownPane = restoreMenuScrollPosition(findMenuContainerInsideTip($thisTip));

                        var $topFilterButtons = $thisTip.find('.resultTopFilter');
                        var $selectedFilter = $topFilterButtons.filter('.selected');

                        if ($selectedFilter.length == 0) { // no filter loaded. load the first one
                            reloadPaneByFilterButton($topFilterButtons.eq(0));
                        } else if ($prevShownPane.length > 0 && paneLoadedToolongAgo($prevShownPane)) {
                            reloadPaneByFilterButton($selectedFilter);
                        }
                    }
                }
            });

            $thisLi.qtip(options);

        });
        
        $body
            .on('mousedown', '.resultTopFilter', function () { // a or div are possible
                var $filterButton = $(this);
                $filterButton.siblings().removeClass('selected').end().addClass('selected');

            });

        if (getQueryString('sidebar') == "1") // from tag approval notice
            setTimeout(function () { $('#extrabarOpener').trigger(_mobileAwareEventName); }, 1000);   
    }

    function registerEventsForTopNavContent($topNavMenu) {
        var viewMoreE = cMemberInfo.isMobileDevice ? 'mousedown' : 'mousedown'; // any touch event will slow down page scroll

        $topNavMenu
        .on(viewMoreE, '.resultTopFilter', function () { // a or div are possible, need to skip a few things when not in topnav
            var $filterButton = $(this);
            var topFilterType = $filterButton.parent().data('topfiltertype');

            if (topFilterType) { // inside topnav as filter

                var filter = $filterButton.data('filter');

                var menuContentSel = '.topnav-menuContent.' + topFilterType;
                var paneID = topFilterType + '-' + filter;
                var $pane = $('#' + paneID);

                if ($pane.length == 0) { // filter content not loaded yet
                    $pane = $('<div id="' + paneID + '" class="' + _paneClass + '" />').appendTo($(menuContentSel + ' .scrollingPane'));

                    if (filter != 'SearchResult') ajaxLoadPane($pane, filter);

                } else if (paneLoadedToolongAgo($pane)) { // filter loaded 5 min ago
                    if (filter != 'SearchResult') ajaxLoadPane($pane, filter);

                } else {
                    scrollPaneIntoView($pane);
                }
            }

        })
        .on(viewMoreE, 'div.topnav-extraLinks.posts span.reload', function () {

            var $thisLink = $(this);
            var $filterButton = findSelectedFilterButtonByContent($thisLink);

            $thisLink.css({ cursor: 'progress' });
            setTimeout(function () { $thisLink.css({ cursor: 'pointer' }); }, 500);

            reloadPaneByFilterButton($filterButton);

            return false;
        })
        .on(viewMoreE, 'div.topnav-extraLinks.posts a.viewmore', function () {

            var $thisLink = $(this);
            var $filterButton = findSelectedFilterButtonByContent($thisLink);
            var filter = $filterButton.data('filter');

            if (filter == 'RecentVisits') {
                qtip.notice($thisLink, ln.NoDataWarning);
                return false;
            }

            $thisLink.attr('href', cPathInfo.ForumDir + filter);
        })
        .on(viewMoreE, 'div.topnav-extraLinks.blog a.viewmore', function () {

            var $thisLink = $(this);

            $thisLink.attr('href', cPathInfo.BlogRootDir);

        })
        .on(viewMoreE, 'div.topnav-extraLinks.pm a.viewmore', function () {

            var $thisLink = $(this);
            var $filterButton = findSelectedFilterButtonByContent($thisLink);
            var filter = $filterButton.data('filter');

            $thisLink.attr('href', cPathInfo.ForumDir + 'pm.aspx' +
                (filter == 'pminbox' ? '?FolderID=0' : ''));

        })
        .on(viewMoreE, 'div.topnav-extraLinks.gallery a.viewmore', function () {

            var $thisLink = $(this);
            var $filterButton = findSelectedFilterButtonByContent($thisLink);
            var filter = $filterButton.data('filter');

            $thisLink.attr('href', cPathInfo.ForumDir + 'gallery.aspx?scrolltophoto=true&myfavorites=' +
                (filter == 'recentphotos' ? 'false' : 'true'));

        });        
    }

    function GetHiddenTopMenuContent($thisLi) { 
        //only called when the qtip is first rendered. content defined in page.
        var $thisLiTopmenu = $thisLi.find('div.topnav-menu');

        if ($thisLiTopmenu.length == 0) return '';

        var thisFilterType = getFilterTypeFromLi($thisLi);
        
        if (thisFilterType == 'pm' && !$thisLi.find('.unreadPMHead')[0]) {
            $thisLi.find('.resultTopFilter:first').remove();
        }

        var filter = $thisLi.find('.resultTopFilter:first').data('filter');
        var indSectionId = thisFilterType + '-' + filter;

        var $menuContentDiv = $('<div />').addClass('topnav-menuContent ' + thisFilterType);
        
        $menuContentDiv.html('<div class="scrollingPane">' +
            '<div id="' + indSectionId + '" class="' + _paneClass + '"></div></div>');

        $thisLiTopmenu.detach().append($menuContentDiv);
        
        var finalContent = $thisLiTopmenu.html();
        
        //content finally stored in tip.content.text
        return finalContent;
    }

    function setMenuDivsWidth($qtip) { // when filter content loaded / shown, we need to set its width for successful scrolling effect
        var $subDivs = $qtip.find('.' + _paneClass);
        var qtipWidth = $qtip.children(qtip.contentSelector).width() - 2;
        $subDivs.css('width', qtipWidth.toString() + 'px'); // standardize all content divs width;
    }

    function setMenuDivHeight($pane) {
        var $qtip = findQtipByContent($pane);
        var qtipHeight = $qtip.find('.topnav-menuTopFilters').innerHeight() + $pane.innerHeight();

        var $content = $qtip.children(qtip.contentSelector);

        if (getTopNavDropDownContentMaxHeight() < qtipHeight) {
            $content.css('overflow', 'auto');
        }
        else {
            $content.css('overflow', 'hidden');
        }
        
        $content.animate({ 'height': (qtipHeight).toString() + 'px' }, 400, forumThemeInfo.jQueryEasing);
    }

    function restoreMenuScrollPosition($menuContent) { // when window is resized and then filter menu reopened
        var storedScrollToEle = $menuContent.data('scrollTo');
        
        if (storedScrollToEle) {
            $menuContent.scrollTo(storedScrollToEle, {duration:0});
        }

        var $scrollTo = $(storedScrollToEle);
        setMenuDivHeight($scrollTo);
        
        return $scrollTo;
    }

    function paneLoadedToolongAgo($pane) {
        return ($pane.data('loaded') < (new Date().getTime() - 3E5)); // 5 min
    }
    
    function reloadPaneByFilterButton($filterButton) { //for reloading a filter
        var filter = $filterButton.data('filter');
        var topfiltertype = $filterButton.parent().data('topfiltertype');
        var $paneToLoad = $('#' + topfiltertype + '-' + filter);

        ajaxLoadPane($paneToLoad, filter, $filterButton.data('searchID'));

        $filterButton.addClass('selected');        
    }

    function ajaxLoadPane($pane, filter, searchID) { //the actual ajax loading of a filter

        var urlToLoad = cPathInfo.ForumDir + 'ws/TopNav.aspx?' +
            (filter == 'SearchResult' ? 's=' + searchID.toString() : 't=' + filter) +
                ' #navdata .newActiveRepeaterWrap';
        
/*        var $loadingPane =
            $('<div />')
            .css({ 'position': 'absolute', 'left': '10px', 'top': '10px' })
            .text(ln.loadingDesc);*/

        var $filterBtnDiv = $pane.closest('.topnav-menuContent').prev();
        $filterBtnDiv.css('opacity', 0.4);
        
        //$pane.after($loadingPane);

        $pane.load(urlToLoad, function (r) {
            $filterBtnDiv.css('opacity', 1);
            //$loadingPane.detach();
            trackThreadRead($pane.selector);

            var $qtip = findQtipByContent($pane);

            setMenuDivsWidth($qtip);
            performDateFormat(false);

            $pane.data('loaded', new Date().getTime()); // for auto reload later on after 5 min

            if (searchID) {
                var $theFilterButton = findTipContainerByContent($pane).find('.resultTopFilter[data-filter=SearchResult]');
                $theFilterButton.show().data('searchID', searchID); // for reload & view more links
            }

            scrollPaneIntoView($pane);
        });
       
    }

    function scrollPaneIntoView($pane) { // scroll the desired filter content
        findMenuContainerByPane($pane)
            .scrollTo($pane,
                { duration: 800, easing: forumThemeInfo.topNavScrollEasing, 
                    onAfter: function (scrollToEle) {$(this).data('scrollTo', scrollToEle); // store the last scrolled To pane
            }
        });
        
        setMenuDivHeight($pane);
    }

    function findMenuContainerByPane($pane) {
        return $pane.closest('.topnav-menuContent');
    }
    
    function findMenuContainerInsideTip($qtip) {
        return $qtip.find('.topnav-menuContent');
    }
    
    function findQtipByContent($pane) {
        return $pane.closest('.qtip');
    }
    
    function findTipContainerByContent($anyContent) {
        return $anyContent.closest(qtip.contentSelector);
    }

    function findSelectedFilterButtonByContent($anyContent) {
        return findTipContainerByContent($anyContent).find('.resultTopFilter.selected');
    }

    function getFilterTypeFromLi($thisLi) {
        return $thisLi.data('filtertype');
    }

    function injectHiddenTopMenu() {
        var menulis = $('#hiddenTopMenu').children('li').toArray().sort(function (a, b) {
            return $(a).data('listindex') - $(b).data('listindex');
        });

        var $primarymenu = $('ul.topnavTabList').children('li');
        
        for (var i = 0; i < menulis.length; i++) {
            var $li = $(menulis[i]);
            $li.insertBefore($primarymenu.eq($li.data('listindex') - i));
        }

    }
/* HTML structure:
-----------------
    qTip for the respective top LI

        qtip-content area

            menuTopFilters w/ data-topfiltertype matching the top LI
                - filter button (w/ a matching pane w/ data-filter specified)
                - filter button
                - etc.

            menu-content
                horizontal scrolling pane holder
                    - pane (w/ a matching filter button)
                    - pane
                    - etc.
*/

$doc.ready(function () {

    var $extrabarOpener = $('#extrabarOpener');

    if ($extrabarOpener.length != 0) {
        
        var options = $.extend(true, getTopNavDropDownTipOptions(), {
            style: { width: 400 },
            events: {
                render: function (e, api) {
                    var $thistip = $(this);
                    var $content = $thistip.children(qtip.contentSelector);
                    $content
                        .css({ 'overflow': 'auto' })
                        .load($extrabarOpener.data('extrabar'), function () {
                            trackThreadRead();
                            performDateFormat(false);

                            //if (!cMemberInfo.isMobileDevice) 
                            produceTagCloud();

                            bindTagApprovalLinks(api.elements.target);

                            setupTopFilterSelect();
                        });
                },
                visible: function () {
                    var $thistip = $(this);
                    var $content = $thistip.children(qtip.contentSelector);
                    var maxHeight = getTopNavDropDownContentMaxHeight();
                    $content.css({ 'max-height': maxHeight });
                }
            }
        });

        $extrabarOpener.bind(_mobileAwareEventName, function () {
            $extrabarOpener.qtip(options);
        });
    } else {
        $extrabarOpener.hide();
    }
});


function setupTopFilterSelect() {
    $('.sidebarTopStatsFilter').off('mousedown', 'a').on('mousedown', 'a', function () {
        var $link = $(this);
        var filter = $link.data('filter');

        if (filter == null) {
            var $selected = $link.siblings('.selected');
            var selFilter = $selected.data('filter');

            if (!selFilter) return true;

            $link.attr('href', cPathInfo.ForumDir + 'Stats.aspx?t=' + selFilter);
            return true;
        }

        var thefid = (typeof currentForumID != 'undefined' ? currentForumID : 0);
        var $stats = $('#topStats');

        $stats.fadeTo('fast', 0.4, function () {
            $stats.load(cPathInfo.ForumDir +
                String.format('ws/extrabar.aspx?c=true&fid={0}&range={1} #topStats', thefid, filter), null, function () {
                    $stats.fadeTo('fast', 1);
                });
        });

    });
}

function bindTagApprovalLinks($tiptarget) {
    $('.sidebarTagAdditional').on('click', 'a.tagapprovelink', function () {
        var $link = $(this);
        var wsName;

        switch ($link.attr('id')) {
            case 'approveTag':
                wsName = 'ApproveTags';
                break;
            case 'disapproveTag':
                wsName = 'DisapproveTags';
                break;
            default:
                wsName = 'DisapproveAndDeleteTags';
        }

        var checkedBoxes = [];

        var sidebarCheckboxesSel = '.sidebarTagCheckboxes';

        var $checkboxesDiv = $(sidebarCheckboxesSel);
        $checkboxesDiv.find('input[type=checkbox]:checked').each(function () {
            checkedBoxes[checkedBoxes.length] = parseInt($(this).val());
        });

        if (checkedBoxes.length == 0) return false;

        JQCallWebService('ws/Tag.aspx/' + wsName, { tagIDs: checkedBoxes }, function () {
            $checkboxesDiv.load($tiptarget.data('extrabar') + ' ' + sidebarCheckboxesSel);
        });

        return false;
    });
}

function produceTagCloud() {
    var $tagArea = $('#sidebarTagsList');
    var $tagitems = $tagArea.find('.tagitem');
    
    var needsUpdate = false;

    var tagCanvasOptions = {
        textColour: '#444',
        outlineColour: '#DADADA',
        shadow: '#888',
        shadowBlur: 2,
        shadowOffset: [0, 1],
        reverse: true,
        weight: true,
        weightFrom : 'data-weight',
        maxSpeed : 0.02,
        wheelZoom: false
    };

    var $canvas = $('#tagCanvas');

    if ($tagitems.length == 0) {
        needsUpdate = true;
    } else {
        if ($tagitems.length == 1 && $tagitems.eq(0).data('weight') == 0) {
            $tagitems.text(ln.NoDataWarning).attr('href', '');
        } else {
            $canvas.show().tagcanvas(tagCanvasOptions, 'tagLinks');
        }
    }

    $tagitems.each(function () {
        //consoleLog(new Date($(this).data('lastgen')));
        if (new Date().getTime() > new Date($(this).data('lastgen')).addDays(1).getTime()) {
            needsUpdate = true;
            return false;
        }
        return true;
    });

    if (needsUpdate) {
        setTimeout(function () {
            JQCallWebService('ws/Tag.aspx/UpdateForumTagStats',
            { forumID: $canvas.data('forumid') }, null, JQOnCallError);            
        }, 1000);
    }
}

$doc.ready(function () {
    var $forumMenuOpener = $('#forumMenuOpener');

    if ($forumMenuOpener.length != 0) {

        var options = $.extend(true, getTopNavDropDownTipOptions(), {
            style: { width: 'auto' },
            events: {
                render: function (e, api) {
                    var $thistip = $(this);
                    var $content = $thistip.children(qtip.contentSelector);
                    $content
                        .css({ 'overflow': 'auto' })
                        .load(cPathInfo.ForumDir + 'ws/TopNav.aspx?t=forums #navdata', function () {
                            $content.find('ul.allforums').find('li.forum').filter(':has(ul)') // the subscription forums and li without childrend ul don't need arrow
                                .append(' <span class="ui-icon ui-icon-triangle-1-e"></span>');
                        });

                    $body.on('mouseenter', '.forumTopMenu li.forum', function () {

                        var $li = $(this);

                        var $childforumLinks = $li.children('ul');

                        if ($childforumLinks.length == 0) return false;

                        var existingQtip = $li.qtip('api');

                        if (existingQtip) {
                            existingQtip.show();
                            return false;
                        }

                        var $allLinksHTML = $('<div />').append(
                            $childforumLinks.clone().addClass('forumTopMenu').removeClass('none'));

                        var finalOptions = $.extend(true, {}, getTopNavDropDownTipOptions(), {
                            //overwrite: true, do not destroy.
                            content: { text: $allLinksHTML.html() },
                            //style: { classes: ' qtip-shadow' },
                            position: { target: $li, my: 'left top', at: ($li.data('depth') > 2 ? 'bottom left' : 'right top'), adjust: { x: -15, y: -5} },
                            show: { delay: 200 },
                            hide: { event: 'mouseleave unfocus', delay: 100, inactive: false },
                            events: {
                                visible: function (ae, aapi) {
                                    var $tip = $(this);

                                    $tip.on('mousedown', 'li', function (eee) {
                                        // necessary to prevent leaving the parent tip visible (needed for the unfocus hide event)
                                        // and to make a clickable.
                                        eee.stopImmediatePropagation();
                                    });

                                    aapi.focus();
                                }
                            }
                        });

                        $li.qtip(finalOptions);

                        return false;
                    });
                },
                visible: function () {
                    var $thistip = $(this);
                    var $content = $thistip.children(qtip.contentSelector);
                    var maxHeight = getTopNavDropDownContentMaxHeight();
                    $content.css({ 'max-height': maxHeight });
                }
            }
        });

        $forumMenuOpener.bind(_mobileAwareEventName, function () {
            $forumMenuOpener
                .qtip(options);
            //.addClass('selected');
        });
    } else {
        $forumMenuOpener.hide();
    }

});

$doc.ready(function () {

    registerTopCustomMenu('li.topCustomMenu');
    
    registerTopCustomMenu('#rightOptionTopMenu');

    if (cMemberInfo.isBMan || cMemberInfo.isUMan || cMemberInfo.isFMan)
        registerAdminMenus();

    registerSubCustomMenus();
});

function registerTopCustomMenu(topLiSel) {
    var $allCustomMenus = $(topLiSel);
    
    for (var i = 0; i < $allCustomMenus.length; i++) {
        
        var $currTopMenu = $allCustomMenus.eq(i);

        var $subOptions = $currTopMenu.children('ul');
        
        var isRightAligned = $currTopMenu.hasClass('right');

        if ($subOptions.find('li')[0]) {

            appendPrependArrowForLi($subOptions, isRightAligned);

            var subMenuOptions = $.extend(true, getTopNavDropDownTipOptions(), {
                show: { ready: false },
                style: { width: 'auto' },
                position: (isRightAligned ? { my: 'top right', at: 'bottom right'} : { my: 'top left', at: 'bottom left' }),
                events: {
                    render: function (e, api) {
                        var $target = api.get('show.target');

                        $(this).children(qtip.contentSelector)
                            .css({ 'overflow': 'auto' })
                            .html(String.format('<ul class="{0}">{1}</ul>',
                                ($target.hasClass('right') ? 'rightOptionMenu' : 'leftOptionMenu'), 
                                $target.children('ul').html()));
                        
                    },
                    visible: function () {
                        var $content = $(this).children(qtip.contentSelector);
                        var maxHeight = getTopNavDropDownContentMaxHeight();
                        $content.css({ 'max-height': maxHeight });
                    }
                }
            });

            $currTopMenu.qtip(subMenuOptions);
            
        } else {
            $currTopMenu.hide();
        }    
    }
}

function registerSubCustomMenus() {

    $body.on('mouseenter', 'ul.rightOptionMenu li, ul.leftOptionMenu li', function () {

        var $theLi = $(this);
        var isRightAligned = $theLi.parent().hasClass('rightOptionMenu');

        var $childMenu = $theLi.children('ul');

        if ($childMenu.length == 0) {
            return false;
        }

        var existingQtip = $theLi.qtip('api');

        if (existingQtip) {
            existingQtip.show();
            return false;
        }

        var $allLinksHTML = $('<div />').append(
            $childMenu.clone().addClass((isRightAligned ? 'rightOptionMenu' : 'leftOptionMenu')).removeClass('none')
        );

        var finalOptions = $.extend(true, {}, getTopNavDropDownTipOptions(), {
            //overwrite: true, do not destroy.
            content: { text: $allLinksHTML.html() },
            //style: { classes: ' qtip-shadow' },
            position: {
                target: $theLi.children('a,span').eq(0),
                my: 'top ' + (isRightAligned ? 'right' : 'left'),
                at: ($theLi.data('depth') > 2 ? 'bottom ' + (isRightAligned ? 'left' : 'right') : (isRightAligned ? 'left' : 'right') + ' top'),
                adjust: { x: (isRightAligned ? -20 : 20), y: -20 }
            },
            show: { delay: 200 },
            hide: {
                event: 'mouseleave ' + ((typeof cMemberInfo != 'undefined' && !cMemberInfo.isMobileDevice) ? 'unfocus' : ''),
                delay: 100, inactive: false
            },
            events: {
                visible: function (ae, aapi) {
                    var $tip = $(this);

                    $tip.on(_mobileAwareEventName, 'li', function (eee) {
                        // necessary to prevent leaving the parent tip visible (needed for the unfocus hide event)
                        // and to make a clickable.
                        eee.stopPropagation();
                        eee.stopImmediatePropagation();
                    });

                    aapi.focus();
                }
            }
        });

        $theLi.qtip(finalOptions);

        return false;
    });
}

function registerAdminMenus() {
    var mTitles = [ln.forumMenuSiteCPDesc, 'Pages & Announcements', 'Forum Management', 'User Management', 'Security Related Options',
                                    'Gallery Options', 'System Related Options', 'All Server Messages', 'Software Activation'];

    var mURLs = ['Default.aspx?tabval=site', 'news.aspx?tabval=editnews', 'forummanager.aspx?tabval=forum', 'user.aspx?tabval=user',
        'akismet.aspx?tabval=akismet', 'gallery.aspx?tabval=gallery', 'maintenance.aspx?tabval=maintain', 'allsrvmsg.aspx?tabval=allsrvmsg',
        'activation.aspx?tabval=activate'];

    var subTitleUrls = {
        admin0: { titles: ["Site Parameters", "Email Settings", "Theme & Display Options", "Home, Forums, Menu & Breadcrumb", "Posts, List View & Search Display", "Default Posting Options", 
            "Social Integration / Sharing", "Search Engine Optimization", "Upload & Download Processing", "Mobile Device Detection", "Valid RegEx Patterns"],
            urls: ["Default.aspx?tabval=site", "email.aspx?tabval=email", "theme.aspx?tabval=theme", "homepage.aspx?tabval=home", "mview.aspx?tabval=mview", "posting.aspx?tabval=posting", 
                "social.aspx?tabval=social", "seo.aspx?tabval=seo", "upload.aspx?tabval=upload", "mobile.aspx?tabval=mobile", "regex.aspx?tabval=regex" ]
        },
        admin2: { titles: ["Forum Management", "Moderator & Group Permissions", "Blogging Integration", "Google Authorship", "Subscription Options", 
                "Thread Labels", "Post Tagging", "PGDCode (BBCode)", "Smilies / Emoticons", "Delete Posts", "Related Server Messages"],
            urls: ["forummanager.aspx?tabval=forum", "forumpermission.aspx?tabval=permission", "blog.aspx?tabval=blog", "authorship.aspx?tabval=author", "subscription.aspx?tabval=subscription", 
                "topictype.aspx?tabval=topictype", "tags.aspx?tabval=tags", "pgdcode.aspx?tabval=pgdcode", "smiley.aspx?tabval=smiley", "mdelete.aspx?tabval=prune", "fsrvmsg.aspx?tabval=fsrvmsg"]
        },
        admin3: { titles: ["User Management", "Registration / Login", "Custom Registration Fields", "User Groups", "Private Message (PM)",
                "User Profile & Member List", "Stock Avatar Upload", "Delete User", "User Rankings", "Bot Detection", "Create Mailing List", "Batch Member Import"],
            urls: ["user.aspx?tabval=user", "reg.aspx?tabval=reg", "customreg.aspx?tabval=customreg", "ugroup.aspx?tabval=group", "pm.aspx?tabval=pm", "profile.aspx?tabval=profile",
                "avatarupload.aspx?tabval=avatar", "udelete.aspx?tabval=udelete", "urank.aspx?tabval=rank", "bot.aspx?tabval=bot", "maillist.aspx?tabval=maillist",
                "memimport.aspx?tabval=memimport"]
        },
        admin4: { titles: ["Akismet Spam Filter", "Report Tickets", "Post Flagging & Auto Ban", "Bad words, IP and Name Filters", "Captcha & ReCaptcha", "Link & PM Spam Prevention"],
            urls: ["akismet.aspx?tabval=akismet", "modticket.aspx?tabval=moderator", "warn.aspx?tabval=warn", "forumfilter.aspx?tabval=filters", "captcha.aspx?tabval=captcha", "linkpmspam.aspx?tabval=lpspam"]
        },
        admin6: { titles: ["Custom Stats", "Basic Maintenance", "Admin / Moderator Log", "Mail / Async Tasks Log", 
            "Error Log", "Scheduled Tasks", "Server Checker", "Config Editor"],
            urls: ["stats.aspx?tabval=stats", "maintenance.aspx?tabval=maintain", "logs.aspx?tabval=logs", "maillog.aspx?tabval=mlog",
                "errorlog.aspx?tabval=error", "schedule.aspx?tabval=sch", "serverchecker.aspx?tabval=srvcehck", "configedit.aspx?tabval=config"]
        }
    };

    var $rightTopMenu = $('#rightOptionTopMenu');
    var $rootAdminLi = $rightTopMenu.find('li.admin');
    
    function addItem($ItemsUL, $cli, $ca) {
        $ItemsUL.append($cli.append($ca));
    }
    
    var $ul = $('<ul class="none" />');
    
    for (var i = 0; i < mTitles.length; i++) {

        if (!cMemberInfo.isBMan) {
            if (cMemberInfo.isUMan && cMemberInfo.isFMan) {
                if (i != 2 && i != 3) continue;
            }
            else if (cMemberInfo.isUMan && i != 3) {
                continue;
            }
            else if (cMemberInfo.isFMan && i != 2) {
                continue;
            }
        }
        
        var $li = $('<li class="admin" />').data('depth', 1).addClass('admin' + i.toString());
        var $a = $('<a />').text(mTitles[i]).attr('href', cPathInfo.ForumDir + 'admincp/' + mURLs[i]);

        if (i == (mTitles.length - 1)) { // activation
            if (cMemberInfo.memID == 0) addItem($ul, $li, $a);
        } else {
            addItem($ul, $li, $a);
        }

    }

    for (var key in subTitleUrls) {
        var $subLi = $ul.find('li.' + key);
        
        if ($subLi.size() == 1) {
            
            var $subsubUL = $('<ul class="none" />');
            
            for (var ii = 0; ii < subTitleUrls[key].titles.length; ii++) {

                if ((key == 'admin6' && ii == 0) &&
                    !(cMemberInfo.memID == 0) && 
                    !(cMemberInfo.isBMan && !cFeatureInfo.isDemoMode)) continue; // custom stats

                if ((key == 'admin6' && ii == 7) &&
                    !(cMemberInfo.memID == 0)) continue; // config editor

                var $subsubLi = $('<li />').data('depth', 2);
                var $subsuba = $('<a />')
                    .text(subTitleUrls[key].titles[ii])
                    .attr('href', cPathInfo.ForumDir + 'admincp/' + subTitleUrls[key].urls[ii]);
                
                addItem($subsubUL, $subsubLi, $subsuba); 
            }

            $subLi.append($subsubUL);
        }
    }

    $rootAdminLi.append($ul);

    appendPrependArrowForLi($rightTopMenu, true);
    
}

function appendPrependArrowForLi($container, isRightAligned) {
    var iconHtml = '<span class="ui-icon ui-icon-triangle-1-{0}">';
    
    $container.find('li:has(ul)').each(function () {
        if (isRightAligned) {
            $(this).prepend(String.format(iconHtml, 'w'));
        }
        else {
            $(this).append(String.format(iconHtml, 'e'));
        }
    });
}

/*splitbutton_key*/

function registerPostbutton() {

    var additionalOptions = {
        style: { tip: false },
        hide: { delay: 600, event: 'unfocus' },
        show: { effect: false},
        events: {
            render: function(e, api) {
                var $tip = $(this);
                $tip.find("a").click(function() { api.hide(); });
            }
        }
    };

    function menuOpener(forSplit) {
        var $link = $(this);

        if ($link.data('menuusesecondary')) {
            $link.next().trigger('mousedown');
            return false;
        }

        var menu = eval($link.data('menu'));

        if ($.isArray(menu))    {

            var menustyle = $link.data('menustyle');
            var menuclasses = {};

            if (menustyle) {
                menuclasses = { style: { classes: menustyle} };
            }

            var menuUpExpand = $link.data('menuupexpand');
            var expandDirection = menuUpExpand ? ['bottom', 'top'] : ['top', 'bottom'];
            var menuXAlign = forSplit ? ' right' : ' left';

            qtip.notice($link,
                resolveAdditionalLinks(menu, forSplit),
                $.extend(true, {}, additionalOptions,
                    { position: { my: expandDirection[0] + menuXAlign, at: expandDirection[1] + menuXAlign,
                        adjust: { x: (forSplit ? 2 : -2), y: (menuUpExpand ? -2 : 2) }
                    }
                }, menuclasses));
            return false;    
        }

        return true;        
    }

    $body
        .off('mousedown', '.splitsecondary')
        .on('mousedown', '.splitsecondary', function () { return menuOpener.call(this, true); })
        .on('click', '.splitsecondary', function () { return $(this).hasClass('postButtonDropdown'); })
        .off('mousedown', '.splitprimary')
        .on('mousedown', '.splitprimary', function () { return menuOpener.call(this, false); })
        .on('click', '.splitprimary', function () {
            var $lnk = $(this);
            return $lnk.data('menu') === '' && $lnk.data('menuusesecondary') !== true;
        });

    }

function resolveAdditionalLinks(additionalLinks, fromsplit) {

    var val = '<ul class="noneList margin0">';
    
    for (var i = 0; i < additionalLinks.length; i++) {
        var link = additionalLinks[i];
        val += '<li>' + String.format('<a class="postButtonDropdown {2}" href="{0}">{1}</a></li>',
            link.url, link.title, (fromsplit ? 'splitsecondary' : ''));
    }

    return val + '</ul>';
}

registerPostbutton();


/*ForumHeaderJs_key*/
 /* Category Show/Hide Functions */

function CatStateToggler(theid) {

    ToggleCatState(theid, true);
    
    return false;
}

function ToggleCatState(catId, animated) {

    var $CatTable = $('#' + catId + '_mainTable');
    
    if (!animated) {
        $CatTable.hide();
        AfterCatToggle(catId);
    } else {
        $CatTable.slideToggle('fast', forumThemeInfo.jQueryEasing, 
            function () { AfterCatToggle(catId); });
    }
    
}

function AfterCatToggle(catId) {

    var $CatTable = $('#' + catId + '_mainTable');
    var isVisible = $CatTable.is(':visible');

    if ($CatTable.length == 0) return;

    $('#' + catId + '_img').attr('src', pageThemeImageURL +
            (isVisible ? ImageCloseFile : ImageOpenFile));

    RecordCatState(catId, isVisible);
}

var _catStateCookie = 'catState';

function RecordCatState(catId, state) {

    var currCookieValue = $.storage.get(_catStateCookie);
    
    if (currCookieValue=="null" || currCookieValue== null) currCookieValue = "";

    currCookieValue = currCookieValue.replace("|" + catId, "").replace("null", "");

    if (!state) currCookieValue += "|" + catId;

    $.storage.set(_catStateCookie, currCookieValue);
}

function RestoreCatState(){
    var currCookieValue = $.storage.get(_catStateCookie);

    if (currCookieValue=="" || currCookieValue=="null" || currCookieValue== null) return;
    
    var arrCurrCookieValue = currCookieValue.split("|");

    if (typeof pageThemeImageURL != 'undefined') {
        for (var i = 0; i < arrCurrCookieValue.length; i++) {

            if (arrCurrCookieValue[i] == '') continue;

            ToggleCatState(arrCurrCookieValue[i], false);

        }
    }
}
/*cssHackEVGA_CLASSIC_V2_key*/

var $_letteringDivSel;
$doc.ready(function () {
    $('html.ie7 table.maintable').attr('cellspacing', '1');

    $_letteringDivSel = $('div.ForumHeaderRow');
    $_letteringDivSel.find('.head').lettering();
    $_letteringDivSel.removeClass('hidden');

});

$win.load(function () {
    if (cMemberInfo.isMobileDevice) return;
    $('ul.topnavTabList').lavaLamp({ easing: 'easeOutBack' });
});

forumThemeInfo.forumIcons.forumUnRead = 'document_copies.png';
forumThemeInfo.forumIcons.forumRead = 'document_empty.png';
forumThemeInfo.forumIcons.forumLink = 'document_redirect.png';
forumThemeInfo.forumIcons.subforumRead = 'blank.gif';
forumThemeInfo.forumIcons.subforumUnRead = 'document_copies_small.png';
forumThemeInfo.forumIcons.subforumLink = 'document_redirect_small.png';
forumThemeInfo.forumIcons.threadUnRead = 'page.png';
forumThemeInfo.forumIcons.threadRead = 'page_white.png';
forumThemeInfo.forumIcons.threadNewestArrow = 'newestmsg.gif';
forumThemeInfo.forumIcons.threadLatestArrow = 'latestmsg.gif';

forumThemeInfo.forumIcons.subforumClose = 'bullet_toggle_minus.png';
forumThemeInfo.forumIcons.subforumOpen = 'bullet_toggle_plus.png';

$.fn.qtip.defaults.style.classes = 'qtip-youtube qtip-rounded qtip-shadow';

if (typeof $.elastislide != 'undefined') {
    $.extend(true, $.elastislide.defaults, {
        imageW: 80,
        minItems: 3,
        border: 4,
        margin: 10,
        easing: forumThemeInfo.jQueryEasing    
    });
}


document.write('<script type="text/javascript" src="' + cPathInfo.ForumDir + 'js/jqueryplugins/jquery.lavalamp.js"><\/script>');
document.write('<script type="text/javascript" src="' + cPathInfo.ForumDir + 'js/jqueryplugins/jquery.lettering.min.js"><\/script>');
