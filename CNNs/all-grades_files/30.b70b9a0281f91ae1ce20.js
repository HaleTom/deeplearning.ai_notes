(window.webpackJsonp=window.webpackJsonp||[]).push([[30],{"AHi/":function(module,t,n){"use strict";n.r(t),n.d(t,"default",function(){return w});var o=n("pVnL"),e=n.n(o),a=n("VbXa"),r=n.n(a),u=n("17x9"),i=n.n(u),c=n("sbe7"),p=n.n(c),s=n("w/1P"),d=n.n(s),l=n("hS5U"),m=n.n(l),g=n("VU1B"),b=n("VGgr"),h=n("BHjd"),f=g.a.create({SignoutForm:{width:"100%",height:"100%",padding:0}}),w=function(t){function SignoutButton(){for(var n,o=arguments.length,e=new Array(o),a=0;a<o;a++)e[a]=arguments[a];return(n=t.call.apply(t,[this].concat(e))||this).state={componentDidMount:!1},n}r()(SignoutButton,t);var n=SignoutButton.prototype;return n.componentDidMount=function componentDidMount(){this.setState(function(){return{componentDidMount:!0}})},n.render=function render(){var t=this.props,n=t.mobileOnly,o=t.onKeyDown,a=t.tabIndex,r=t.targetRef,u=this.state.componentDidMount,i=b.a.get("CSRF3-Token"),c=d()("c-ph-right-nav-button","rc-HeaderRightNavButton",n&&"c-ph-right-nav-mobile-only"),s=u&&Object(h.a)("logout",i)||"";return p.a.createElement("li",{className:c},p.a.createElement("form",e()({},Object(g.d)("c-ph-right-nav-button",f.SignoutForm),{action:s,method:"post"}),p.a.createElement("button",{id:"logout-btn",role:"menuitem",tabIndex:a,ref:r,onKeyDown:o,className:"sign-out",type:"submit","data-popup-close":!0,style:{border:"none"}},m()("Log Out"))))},SignoutButton}(c.Component);w.propTypes={mobileOnly:i.a.bool,tabIndex:i.a.number,onKeyDown:i.a.func,targetRef:i.a.func}}}]);
//# sourceMappingURL=30.b70b9a0281f91ae1ce20.js.map